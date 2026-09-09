// RUN: split-file %s %t
// RUN: triton-opt %t/assign.mlir -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck %t/assign.mlir
// RUN: triton-opt %t/circular.mlir -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase --cse | FileCheck %t/circular.mlir
// RUN: triton-opt %t/from-insert.mlir -split-input-file --allow-unregistered-dialect --nvws-insert-semas --nvws-semaphore-optimize --nvws-assign-semaphore-stage-phase --cse | FileCheck %t/from-insert.mlir

//--- assign.mlir

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  tt.func @assign_stage_basic(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // Phase flip BEFORE acquire
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[BUF]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[BUF]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %val = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      ttg.local_store %val, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @mixed_initial_released_mask
  tt.func @mixed_initial_released_mask() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.create %{{.*}} released = 5
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.constant -6 : i32
    %sem = nvws.semaphore.create %buf released = 5 : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @split_phase_shared_across_partitions
  // CHECK: [[SEM:%.*]] = nvws.semaphore.create
  // CHECK: {{%.*}}:4 = scf.for {{.*}} iter_args([[STAGE_IN:%[^ ]+]] = {{%[^,]+}}, [[L0_IN:%[^ ]+]] = {{%[^,]+}}, [[L1_IN:%[^ ]+]] = {{%[^,]+}}, {{%[^)]+}}) -> (i32, i32, i32, i32)
  // CHECK: [[X_SHIFT:%.*]] = arith.shli {{%.*}}, [[X_STAGE:%.*]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[P1_L0:%.*]] = arith.xori [[L0_IN]], [[X_SHIFT]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[X_PHASE_SHIFT:%.*]] = arith.shrui [[P1_L0]], [[X_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[X_PHASE:%.*]] = arith.andi [[X_PHASE_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: nvws.semaphore.acquire [[SEM]][[[X_STAGE]], [[X_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // CHECK: [[Y_SHIFT:%.*]] = arith.shli {{%.*}}, [[Y_STAGE:%.*]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[P2_L0:%.*]] = arith.xori [[P1_L0]], [[Y_SHIFT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[Y_PHASE_SHIFT:%.*]] = arith.shrui [[P2_L0]], [[Y_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[Y_PHASE:%.*]] = arith.andi [[Y_PHASE_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: nvws.semaphore.acquire [[SEM]][[[Y_STAGE]], [[Y_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // CHECK: [[Z_SHIFT:%.*]] = arith.shli {{%.*}}, [[Z_STAGE:%.*]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: [[P1_L1:%.*]] = arith.xori [[L1_IN]], [[Z_SHIFT]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: [[Z_PHASE_SHIFT:%.*]] = arith.shrui [[P1_L1]], [[Z_STAGE]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: [[Z_PHASE:%.*]] = arith.andi [[Z_PHASE_SHIFT]], {{%.*}} {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: nvws.semaphore.acquire [[SEM]][[[Z_STAGE]], [[Z_PHASE]]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
  // CHECK: scf.yield {{.*}}, [[P2_L0]], [[P1_L1]], {{%.*}} : i32, i32, i32, i32
  // CHECK: ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1, 2>, array<i32: 1>, array<i32: 1>]
  tt.func @split_phase_shared_across_partitions(%lb: i32, %ub: i32,
                                                 %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = 305 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %base released = 7 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %driver = nvws.semaphore.create %base released = 7 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %zd0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td0 = nvws.semaphore.acquire %driver[%zd0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd0 = nvws.semaphore.buffer %driver[%zd0], %td0 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd0) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %z0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %t0 = nvws.semaphore.acquire %sem[%z0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %zd1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td1 = nvws.semaphore.acquire %driver[%zd1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd1 = nvws.semaphore.buffer %driver[%zd1], %td1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd1) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %z1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %t1 = nvws.semaphore.acquire %sem[%z1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %zd2 = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td2 = nvws.semaphore.acquire %driver[%zd2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd2 = nvws.semaphore.buffer %driver[%zd2], %td2 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd2) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %z2 = arith.constant {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %t2 = nvws.semaphore.acquire %sem[%z2] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 11 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @base_phase_independent_across_disjoint_partitions
  // CHECK: [[BASE_EMPTY:%.*]] = nvws.semaphore.create
  // CHECK: {{%.*}}:3 = scf.for {{.*}} iter_args({{%[^,]+}} = {{%[^,]+}}, [[BASE_P1_IN:%[^ ]+]] = {{%[^,]+}}, [[BASE_P2_IN:%[^ ]+]] = {{%[^)]+}}) -> (i32, i32, i32)
  // CHECK: [[BASE_A_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[BASE_A_WORD:%.*]] = arith.xori [[BASE_P2_IN]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
  // CHECK: [[BASE_A_SHIFT:%.*]] = arith.shrui [[BASE_A_WORD]], [[BASE_A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
  // CHECK: [[BASE_A_PHASE:%.*]] = arith.andi [[BASE_A_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
  // CHECK: nvws.semaphore.acquire [[BASE_EMPTY]][[[BASE_A_STAGE]], [[BASE_A_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // CHECK-NOT: nvws.semaphore.acquire [[BASE_EMPTY]]
  // CHECK: [[BASE_B_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // CHECK: [[BASE_B_WORD:%.*]] = arith.xori [[BASE_P1_IN]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: [[BASE_B_SHIFT:%.*]] = arith.shrui [[BASE_B_WORD]], [[BASE_B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: [[BASE_B_PHASE:%.*]] = arith.andi [[BASE_B_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
  // CHECK: nvws.semaphore.acquire [[BASE_EMPTY]][[[BASE_B_STAGE]], [[BASE_B_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // CHECK-NOT: nvws.semaphore.acquire [[BASE_EMPTY]]
  // CHECK: scf.yield {{.*}}, [[BASE_B_WORD]], [[BASE_A_WORD]] : i32, i32, i32
  // CHECK: ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>, array<i32: 2>]
  tt.func @base_phase_independent_across_disjoint_partitions(
      %lb: i32, %ub: i32, %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 306 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x1xi32, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base released = 15 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %a0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %ta = nvws.semaphore.acquire %empty[%a0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %ba = nvws.semaphore.buffer %empty[%a0], %ta {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%ba) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %b0 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tb = nvws.semaphore.acquire %empty[%b0] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bb = nvws.semaphore.buffer %empty[%b0], %tb {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bb) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    } {tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_next_iter_if_result_use_d
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 3
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
  // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[USE_D:%.*]] = %true, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
  // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], [[USE_D]], %true {ttg.partition = array<i32: 1>}
  // CHECK: [[FLAG:%.*]] = arith.xori %{{.*}}, %true {ttg.partition = array<i32: 0, 1>} : i1
  // CHECK: scf.if {{.*}} -> (!ttg.async.token, i32, i32, i32)
  // CHECK: [[C1_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
  // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[C1_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
  // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
  // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], {{.*}}] {ttg.partition = array<i32: 1>}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], {{.*}}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}}, [[FLAG]], {{.*}}
  tt.func @matmul_tma_acc_with_next_iter_if_result_use_d(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1) : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %useD_next = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_6 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %12, %useD_next : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @shared_stage_two_semaphores
  tt.func @shared_stage_two_semaphores() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM0:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[SEM1:%.*]] = nvws.semaphore.create %{{.*}}
    // CHECK: [[S_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant -4 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    %sem0 = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %sem1 = nvws.semaphore.create %buf : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // Stage advance: addi/cmpi/select wrapping at 2
    // CHECK: [[STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[NEXT:%.*]] = arith.addi [[S_INIT]], [[STEP]] : i32
    // CHECK: [[DEPTH:%.*]] = arith.constant 2 : i32
    // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], [[DEPTH]] : i32
    // CHECK: [[ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[ADV:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT]] : i32
    // Phase flip sem0 (released=3, init=-4), then acquire
    // CHECK: [[C1_0:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH0:%.*]] = arith.shli [[C1_0]], [[ADV]] : i32
    // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE_INIT]], [[SH0]] : i32
    // CHECK: [[PE_SHR:%.*]] = arith.shrui [[PE_NEW]], [[ADV]] : i32
    // CHECK: [[PE_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_BIT:%.*]] = arith.andi [[PE_SHR]], [[PE_C1]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[SEM0]][[[ADV]], [[PE_BIT]]]
    // Phase flip sem1 (released omitted, init=-1), then acquire
    // CHECK: [[C1_1:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH1:%.*]] = arith.shli [[C1_1]], [[ADV]] : i32
    // CHECK: [[PF_NEW:%.*]] = arith.xori [[PF_INIT]], [[SH1]] : i32
    // CHECK: [[PF_SHR:%.*]] = arith.shrui [[PF_NEW]], [[ADV]] : i32
    // CHECK: [[PF_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF_BIT:%.*]] = arith.andi [[PF_SHR]], [[PF_C1]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[SEM1]][[[ADV]], [[PF_BIT]]]
    %tok0 = nvws.semaphore.acquire %sem0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %tok1 = nvws.semaphore.acquire %sem1 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

    // CHECK: [[BUF0:%.*]] = nvws.semaphore.buffer [[SEM0]][[[ADV]]], [[TOK0]]
    // CHECK: ttg.local_store {{%.*}}, [[BUF0]]
    %view0 = nvws.semaphore.buffer %sem0, %tok0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    %v = arith.constant dense<0> : !elt
    ttg.local_store %v, %view0 : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

    // CHECK: nvws.semaphore.release [[SEM0]][[[ADV]]], [[TOK0]] [#nvws.async_op<none>]
    // CHECK: nvws.semaphore.release [[SEM1]][[[ADV]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.semaphore.release %sem0, %tok0 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    nvws.semaphore.release %sem1, %tok1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @multi_result_buffer_unused_sibling
  tt.func @multi_result_buffer_unused_sibling(%lb: i32, %ub: i32, %step: i32) {
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}} released = 3
    %sem = nvws.semaphore.create %buf0, %buf1 released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>, !ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %v = arith.constant dense<0> : !elt
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]]
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>, !ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[VIEW:%.*]]:2 = nvws.semaphore.buffer [[SEM]]
      %view0, %view1 = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>, !ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: ttg.local_store {{.*}}, [[VIEW]]#0
      ttg.local_store %v, %view0 {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>, !ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %buf0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    ttg.local_dealloc %buf1 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @if_observation
  tt.func @if_observation(%cond: i1, %lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // No stage advance here: first use is not provably Store on all paths.
      // Phase flip BEFORE acquire
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[BUF]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[BUF]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

      // scf.if with NO results (no stage/phase threading through if)
      scf.if %cond {
        %x = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
        "use"(%x) {ttg.partition = array<i32: 0>} : (!elt) -> ()
      } {ttg.partition = array<i32: 0>}

      %v = arith.constant {ttg.partition = array<i32: 0>} dense<0> : !elt
      ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @for_body_store_post_loop_load
  tt.func @for_body_store_post_loop_load(%lb: i32, %ub: i32, %step: i32,
                                         %lb1: i32, %ub1: i32, %step1: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // No stage advance here: a store inside the nested loop competes with a
      // later post-loop load of the same buffer lineage.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      // CHECK: "foo_load"([[BUF]]) {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      scf.for %j = %lb1 to %ub1 step %step1 : i32 {
        "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>) -> ()
      } {ttg.partition = array<i32: 0>}
      "foo_load"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared2d_t = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @view_path_observation
  tt.func @view_path_observation(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // The first real access of the semaphore-buffer lineage is foo_store on
      // %view, so this advances stage even though a later alias path loads.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[STAGE_INC:%.*]] = arith.addi [[STAGE]], [[C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[STAGE_INC]], [[C2]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[STAGE_NEW:%.*]] = arith.select [[WRAP]], [[C0]], [[STAGE_INC]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_SHIFT:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1_SHIFT]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE_NEW]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE_NEW]]], [[TOK]] {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>
      %trans = ttg.memdesc_trans %view {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2> -> !ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      // CHECK: "foo_load"(%{{.*}}) {ttg.partition = array<i32: 0>}
      "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>) -> ()
      "foo_load"(%trans) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE_NEW]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE_NEW]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @if_view_fallthrough_store
  tt.func @if_view_fallthrough_store(%cond: i1, %lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[C1_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0_INIT:%.*]] = arith.constant -4 : i32
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C1_INIT]], [[PHASE:%.*]] = [[C0_INIT]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // The branch only creates a view alias; both true and false paths fall
      // through to the same later store, so this still advances stage.
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[STAGE_INC:%.*]] = arith.addi [[STAGE]], [[C1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[STAGE_INC]], [[C2]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      // CHECK: [[STAGE_NEW:%.*]] = arith.select [[WRAP]], [[C0]], [[STAGE_INC]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_SHIFT:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1_SHIFT]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[PHASE_NEW]], [[STAGE_NEW]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[C1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PHASE_BIT:%.*]] = arith.andi [[PHASE_SHR]], [[C1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[STAGE_NEW]], [[PHASE_BIT]]] {ttg.partition = array<i32: 0>}
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[SEM]][[[STAGE_NEW]]], [[TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: "foo_store"([[BUF]]) {ttg.partition = array<i32: 0>}
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>
      scf.if %cond {
        %trans = ttg.memdesc_trans %view {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2> -> !ttg.memdesc<2x2xi32, #shared2d_t, #smem, mutable, 2x2x2>
      } {ttg.partition = array<i32: 0>}
      "foo_store"(%view) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<2x2xi32, #shared2d, #smem, mutable, 2x2x2>) -> ()
      // CHECK: nvws.semaphore.release [[SEM]][[[STAGE_NEW]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[STAGE_NEW]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x2x2xi32, #shared2d, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

#shared9 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem9 = #ttg.shared_memory
#tmem9 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales9 = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @scale_a_buffer_read
  tt.func @scale_a_buffer_read(%arg0: !ttg.memdesc<128x64xf16, #shared9, #smem9>, %arg1: !ttg.memdesc<64x128xf16, #shared9, #smem9>, %arg2: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>) {
    %true = arith.constant true
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY_A:%.*]] = nvws.semaphore.create %arg3 released = 1
    // CHECK: [[FULL_A:%.*]] = nvws.semaphore.create %arg3
    %empty = nvws.semaphore.create %arg3 released = 1 : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    %full = nvws.semaphore.create %arg3 : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    // CHECK: [[TOK_A:%.*]] = nvws.semaphore.acquire [[FULL_A]]
    %tok = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]> -> !ttg.async.token
    // CHECK: [[BUF_A:%.*]] = nvws.semaphore.buffer [[FULL_A]][
    %buf = nvws.semaphore.buffer %full, %tok : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, [[BUF_A]], {{.*}}, %true, %true lhs = e4m3 rhs = e4m3
    ttng.tc_gen5_mma_scaled %arg0, %arg1, %acc, %buf, %arg2, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf16, #shared9, #smem9>, !ttg.memdesc<64x128xf16, #shared9, #smem9>, !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
    // CHECK: nvws.semaphore.release [[EMPTY_A]][
    nvws.semaphore.release %empty, %tok [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @scale_b_buffer_read
  tt.func @scale_b_buffer_read(%arg0: !ttg.memdesc<128x64xf16, #shared9, #smem9>, %arg1: !ttg.memdesc<64x128xf16, #shared9, #smem9>, %arg2: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>) {
    %true = arith.constant true
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY_B:%.*]] = nvws.semaphore.create %arg3 released = 1
    // CHECK: [[FULL_B:%.*]] = nvws.semaphore.create %arg3
    %empty = nvws.semaphore.create %arg3 released = 1 : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    %full = nvws.semaphore.create %arg3 : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>
    // CHECK: [[TOK_B:%.*]] = nvws.semaphore.acquire [[FULL_B]]
    %tok = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]> -> !ttg.async.token
    // CHECK: [[BUF_B:%.*]] = nvws.semaphore.buffer [[FULL_B]][
    %buf = nvws.semaphore.buffer %full, %tok : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, {{.*}}, [[BUF_B]], %true, %true lhs = e4m3 rhs = e4m3
    ttng.tc_gen5_mma_scaled %arg0, %arg1, %acc, %arg2, %buf, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf16, #shared9, #smem9>, !ttg.memdesc<64x128xf16, #shared9, #smem9>, !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory, mutable>
    // CHECK: nvws.semaphore.release [[EMPTY_B]][
    nvws.semaphore.release %empty, %tok [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %ub = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 7
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
    // CHECK: [[C2_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant -8 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    %empty = nvws.semaphore.create %0 released = 7 : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %0 : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[LOOP:%.*]]:4 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C2_INIT]], [[PE:%.*]] = [[PE_INIT]], {{%.*}} = [[PF_INIT]], {{%.*}} = [[PF_INIT]]) -> (i32, i32, i32, i32)
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      // Stage advance: addi/cmpi/select wrapping at 3
      // CHECK: [[STEP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1 : i32
      // CHECK: [[NEXT:%.*]] = arith.addi [[STAGE]], [[STEP]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[C3:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 3 : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], [[C3]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
      // CHECK: [[NEW_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT]] {ttg.partition = array<i32: 0, 1, 2>} : i32
      // Phase flip EMPTY (released=7, init=-8), then acquire
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSHIFT:%.*]] = arith.shli [[PC1]], [[NEW_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE]], [[PSHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_SHR:%.*]] = arith.shrui [[PE_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE_BIT:%.*]] = arith.andi [[PE_SHR]], [[PE_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEW_STAGE]], [[PE_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[NEW_STAGE]]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]][[[NEW_STAGE]]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %token = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers = nvws.semaphore.buffer %empty, %token {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      ttg.local_store %2, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.semaphore.release %full, %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // Consumer1: phase flip FULL, then acquire
      // CHECK: [[GC1_1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSHIFT1:%.*]] = arith.shli [[GC1_1]], [[NEW_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PF1:%.*]], [[GSHIFT1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF1_BIT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[NEW_STAGE]], [[PF1_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL]][[[NEW_STAGE]]], [[GTOK1]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[GBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[NEW_STAGE]]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %token_1 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers_0 = nvws.semaphore.buffer %full, %token_1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %3 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %empty, %token_1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      // Consumer2: phase flip FULL, then acquire
      // CHECK: [[GC1_2:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[GSHIFT2:%.*]] = arith.shli [[GC1_2]], [[NEW_STAGE]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_NEW:%.*]] = arith.xori [[PF2:%.*]], [[GSHIFT2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_SHR:%.*]] = arith.shrui [[PF2_NEW]], [[NEW_STAGE]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[PF2_BIT:%.*]] = arith.andi [[PF2_SHR]], [[PF2_MASK]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]][[[NEW_STAGE]], [[PF2_BIT]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL]][[[NEW_STAGE]]], [[GTOK2]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[GBUF2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[NEW_STAGE]]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      %token_3 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %buffers_2 = nvws.semaphore.buffer %full, %token_3 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %4 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %empty, %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[NEW_STAGE]], [[PE_NEW]], [[PF1_NEW]], [[PF2_NEW]] : i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}

    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_lowering
  tt.func @semaphore_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %f : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %g : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32

    // CHECK: [[E0:%.*]] = nvws.semaphore.create {{%.*}} released = 7
    // CHECK: [[F0:%.*]] = nvws.semaphore.create {{%.*}}
    // CHECK: [[S0_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE0_INIT:%.*]] = arith.constant -8 : i32
    // CHECK: [[PF0_INIT:%.*]] = arith.constant -1 : i32
    // CHECK: [[E1:%.*]] = nvws.semaphore.create {{%.*}} released = 7
    // CHECK: [[F1:%.*]] = nvws.semaphore.create {{%.*}}
    // CHECK: [[S1_INIT:%.*]] = arith.constant 2 : i32
    // CHECK: [[PE1_INIT:%.*]] = arith.constant -8 : i32
    // CHECK: [[PF1_INIT:%.*]] = arith.constant -1 : i32
    %empty0 = nvws.semaphore.create %d, %e released = 7 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %full0 = nvws.semaphore.create %d, %e : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %empty1 = nvws.semaphore.create %f, %g released = 7 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %full1 = nvws.semaphore.create %f, %g : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    // CHECK: [[LOOP:%.*]]:6 = scf.for {{.*}} iter_args([[S0:%.*]] = [[S0_INIT]], [[PE0:%.*]] = [[PE0_INIT]], [[PF0:%.*]] = [[PF0_INIT]], [[S1:%.*]] = [[S1_INIT]], [[PE1:%.*]] = [[PE1_INIT]], [[PF1:%.*]] = [[PF1_INIT]]) -> (i32, i32, i32, i32, i32, i32)
    scf.for %i = %lb to %ub step %c1_i32 : i32{
      // Group0 producer: stage advance, then phase flip pe0 BEFORE acquire E0
      // CHECK: [[PC1_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S0_NEXT_RAW:%.*]] = arith.addi [[S0]], [[PC1_P0]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P0_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[S0_WRAP:%.*]] = arith.cmpi eq, [[S0_NEXT_RAW]], [[P0_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S0_NEXT:%.*]] = arith.select [[S0_WRAP]], [[P0_ZERO]], [[S0_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[PC1_P0B:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSH0:%.*]] = arith.shli [[PC1_P0B]], [[S0_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_NEW:%.*]] = arith.xori [[PE0]], [[PSH0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_SHR:%.*]] = arith.shrui [[PE0_NEW]], [[S0_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE0_BIT:%.*]] = arith.andi [[PE0_SHR]], [[PE0_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PTOK0:%.*]] = nvws.semaphore.acquire [[E0]][[[S0_NEXT]], [[PE0_BIT]]] {ttg.partition = array<i32: 0>}
      %ptok0 = nvws.semaphore.acquire %empty0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
      // CHECK: [[PBUF0:%.*]]:2 = nvws.semaphore.buffer [[E0]][[[S0_NEXT]]], [[PTOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: "op1_store"([[PBUF0]]#0) {ttg.partition = array<i32: 0>}
      // CHECK: "op2_store"([[PBUF0]]#1) {ttg.partition = array<i32: 0>}
      %1:2 = nvws.semaphore.buffer %empty0, %ptok0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
      "op1_store"(%1#0) {ttg.partition = array<i32: 0>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>) -> ()
      "op2_store"(%1#1)  {ttg.partition = array<i32: 0>} : (!ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
      // CHECK: nvws.semaphore.release [[F0]][[[S0_NEXT]]], [[PTOK0]] [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %full0, %ptok0 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

      // Group0 consumer: phase flip pf0 BEFORE acquire F0
      // CHECK: [[GC1_C0:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSH0:%.*]] = arith.shli [[GC1_C0]], [[S0_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_NEW:%.*]] = arith.xori [[PF0]], [[GSH0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_SHR:%.*]] = arith.shrui [[PF0_NEW]], [[S0_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF0_BIT:%.*]] = arith.andi [[PF0_SHR]], [[PF0_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0_NEXT]], [[PF0_BIT]]] {ttg.partition = array<i32: 1>}
      %gtok0 = nvws.semaphore.acquire %full0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
      // CHECK: [[GBUF0:%.*]]:2 = nvws.semaphore.buffer [[F0]][[[S0_NEXT]]], [[GTOK0]] {ttg.partition = array<i32: 1>}
      // CHECK: "op3_load"([[GBUF0]]#0, [[GBUF0]]#1) {ttg.partition = array<i32: 1>}
      %2:2 = nvws.semaphore.buffer %full0, %gtok0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
      "op3_load"(%2#0, %2#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
      // CHECK: nvws.semaphore.release [[E0]][[[S0_NEXT]]], [[GTOK0]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty0, %gtok0 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: [[IFRES:%.*]]:3 = scf.if {{%.*}} -> (i32, i32, i32)
      scf.if %cond {
      } else {
        %ptok1 = nvws.semaphore.acquire %empty1 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
        %4:2 = nvws.semaphore.buffer %empty1, %ptok1 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
        "op4_store"(%4#0, %4#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
        nvws.semaphore.release %full1, %ptok1 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        %gtok1 = nvws.semaphore.acquire %full1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.async.token
        %5:2 = nvws.semaphore.buffer %full1, %gtok1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>
        "op5_load"(%5#0, %5#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared0, #smem, mutable>) -> ()
        nvws.semaphore.release %empty1, %gtok1 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1]], [[PE1]], [[PF1]] : i32, i32, i32
      // CHECK: } else {
      // Else branch advances S1, then flips pe1 BEFORE acquire E1.
      // CHECK: [[EC1_P1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S1_NEXT_RAW:%.*]] = arith.addi [[S1]], [[EC1_P1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[S1_WRAP:%.*]] = arith.cmpi eq, [[S1_NEXT_RAW]], [[S1_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S1_NEXT:%.*]] = arith.select [[S1_WRAP]], [[S1_ZERO]], [[S1_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[EC1_P1B:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ESH_P1:%.*]] = arith.shli [[EC1_P1B]], [[S1_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_NEW:%.*]] = arith.xori [[PE1]], [[ESH_P1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_SHR:%.*]] = arith.shrui [[PE1_NEW]], [[S1_NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PE1_BIT:%.*]] = arith.andi [[PE1_SHR]], [[PE1_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ETOK_P1:%.*]] = nvws.semaphore.acquire [[E1]][[[S1_NEXT]], [[PE1_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[EBUF_P1:%.*]]:2 = nvws.semaphore.buffer [[E1]][[[S1_NEXT]]], [[ETOK_P1]] {ttg.partition = array<i32: 0>}
      // CHECK: "op4_store"([[EBUF_P1]]#0, [[EBUF_P1]]#1) {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[F1]][[[S1_NEXT]]], [[ETOK_P1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Phase flip pf1 BEFORE acquire F1
      // CHECK: [[FC1_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[FSH_C1:%.*]] = arith.shli [[FC1_C1]], [[S1_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PF1]], [[FSH_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_NEW]], [[S1_NEXT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PF1_BIT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[FTOK_C1:%.*]] = nvws.semaphore.acquire [[F1]][[[S1_NEXT]], [[PF1_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[FBUF_C1:%.*]]:2 = nvws.semaphore.buffer [[F1]][[[S1_NEXT]]], [[FTOK_C1]] {ttg.partition = array<i32: 1>}
      // CHECK: "op5_load"([[FBUF_C1]]#0, [[FBUF_C1]]#1) {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][[[S1_NEXT]]], [[FTOK_C1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1_NEXT]], [[PE1_NEW]], [[PF1_NEW]] : i32, i32, i32
      // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      } {ttg.partition = array<i32: 0, 1>}

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[S0_NEXT]], [[PE0_NEW]], [[PF0_NEW]], [[IFRES]]#0, [[IFRES]]#1, [[IFRES]]#2 : i32, i32, i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 0>, array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 1
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
    // CHECK: [[S_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant -2 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    // Pre-loop: stage advance, phase-bitset toggle and extraction, acquire EMPTY
    // CHECK: [[SA_STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_ADD:%.*]] = arith.addi [[S_INIT]], [[SA_STEP]] : i32
    // CHECK: [[SA_DEPTH:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_CMP:%.*]] = arith.cmpi eq, [[SA_ADD]], [[SA_DEPTH]] : i32
    // CHECK: [[SA_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[PRE_STAGE:%.*]] = arith.select [[SA_CMP]], [[SA_ZERO]], [[SA_ADD]] : i32
    // CHECK: [[PE_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_SHIFT:%.*]] = arith.shli [[PE_C1]], [[PRE_STAGE]] : i32
    // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE_INIT]], [[PE_SHIFT]] : i32
    // CHECK: [[PE_SHR:%.*]] = arith.shrui [[PE_NEW]], [[PRE_STAGE]] : i32
    // CHECK: [[PE_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_PRE:%.*]] = arith.andi [[PE_SHR]], [[PE_MASK]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[PRE_STAGE]], [[PE_PRE]]]
    // CHECK: [[BUF_INIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[TOK0]]
    // CHECK: ttng.tmem_store {{%.*}}, [[BUF_INIT]][], {{%.*}}
    %empty = nvws.semaphore.create %result released = 1 : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: scf.for {{.*}} : i32 {
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %4 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[TOK0]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %10 = nvws.semaphore.buffer %empty, %token {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // Post-loop: release FULL, toggle and extract its phase bit, acquire FULL, tmem_load, release EMPTY
    // CHECK: nvws.semaphore.release [[FULL]][[[PRE_STAGE]]], [[TOK0]] [#nvws.async_op<tc5mma>]
    nvws.semaphore.release %full, %token [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[PF_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF_SHIFT:%.*]] = arith.shli [[PF_C1]], [[PRE_STAGE]] : i32
    // CHECK: [[PF_NEW:%.*]] = arith.xori [[PF_INIT]], [[PF_SHIFT]] : i32
    // CHECK: [[PF_SHR:%.*]] = arith.shrui [[PF_NEW]], [[PRE_STAGE]] : i32
    // CHECK: [[PF_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF_POST:%.*]] = arith.andi [[PF_SHR]], [[PF_MASK]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[PRE_STAGE]], [[PF_POST]]]
    // CHECK: [[BUF_POST:%.*]] = nvws.semaphore.buffer [[FULL]][[[PRE_STAGE]]], [[TOK1]]
    // CHECK: ttng.tmem_load [[BUF_POST]][]
    %token_1 = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %full, %token_1 : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %3[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    // CHECK: nvws.semaphore.release [[EMPTY]][[[PRE_STAGE]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.semaphore.release %empty, %token_1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
    // Pre-loop: stage advance, phase-bitset toggles and extraction, acquire EMPTY
    // CHECK: [[S_INIT:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant -4 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    // CHECK: [[SA_STEP:%.*]] = arith.constant 1 : i32
    // CHECK: [[SA_ADD:%.*]] = arith.addi [[S_INIT]], [[SA_STEP]] : i32
    // CHECK: [[SA_DEPTH:%.*]] = arith.constant 2 : i32
    // CHECK: [[SA_CMP:%.*]] = arith.cmpi eq, [[SA_ADD]], [[SA_DEPTH]] : i32
    // CHECK: [[SA_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[PRE_STAGE:%.*]] = arith.select [[SA_CMP]], [[SA_ZERO]], [[SA_ADD]] : i32
    // CHECK: [[PF1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_SHIFT:%.*]] = arith.shli [[PF1_C1]], [[PRE_STAGE]] : i32
    // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PE_INIT]], [[PF1_SHIFT]] : i32
    // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_NEW]], [[PRE_STAGE]] : i32
    // CHECK: [[PF1_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_OUT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] : i32
    // CHECK: [[PE1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE1_SHIFT:%.*]] = arith.shli [[PE1_C1]], [[PRE_STAGE]] : i32
    // CHECK: [[PE1_NEW:%.*]] = arith.xori [[PE_INIT]], [[PE1_SHIFT]] : i32
    // CHECK: [[PE1_SHR:%.*]] = arith.shrui [[PE1_NEW]], [[PRE_STAGE]] : i32
    // CHECK: [[PE1_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE_PRE:%.*]] = arith.andi [[PE1_SHR]], [[PE1_MASK]] : i32
    // CHECK: [[PRETOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[PRE_STAGE]], [[PE_PRE]]]
    // CHECK: [[BUF_INIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[PRE_STAGE]]], [[PRETOK]]
    // CHECK: ttng.tmem_store {{%.*}}, [[BUF_INIT]][], {{%.*}}
    %empty = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:4 = scf.for {{.*}} iter_args([[FTOK:%.*]] = [[PRETOK]], [[FSTAGE:%.*]] = [[PRE_STAGE]], {{%.*}}, {{%.*}}) -> (!ttg.async.token, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[FULL]][[[FSTAGE]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // Consumer: phase-bitset toggle and extraction BEFORE acquire FULL
      // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[GSHIFT:%.*]] = arith.shli [[GC1]], [[FSTAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[FPF_NEW:%.*]] = arith.xori [[FPF:%.*]], [[GSHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GSHR:%.*]] = arith.shrui [[FPF_NEW]], [[FSTAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GMASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[FPF_BIT:%.*]] = arith.andi [[GSHR]], [[GMASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[FSTAGE]], [[FPF_BIT]]] {ttg.partition = array<i32: 0>}
      %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %11 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_LOAD:%.*]] = nvws.semaphore.buffer [[FULL]][[[FSTAGE]]], [[GTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_load [[BUF_LOAD]][] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY]][[[FSTAGE]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_3, %token_4 = ttng.tmem_load %11[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // Stage advance, phase-bitset toggle and extraction, re-acquire EMPTY
      // CHECK: [[NSA_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[NSA_ADD:%.*]] = arith.addi [[FSTAGE]], [[NSA_STEP]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[NSA_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[NSA_CMP:%.*]] = arith.cmpi eq, [[NSA_ADD]], [[NSA_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[NSA_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[NSA_CMP]], [[NSA_ZERO]], [[NSA_ADD]] {ttg.partition = array<i32: 0, 1>}
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PSHIFT:%.*]] = arith.shli [[PC1]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[FPE_NEW:%.*]] = arith.xori [[FPE:%.*]], [[PSHIFT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PSHR:%.*]] = arith.shrui [[FPE_NEW]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PMASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[FPE_BIT:%.*]] = arith.andi [[PSHR]], [[PMASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[FPE_BIT]]] {ttg.partition = array<i32: 1>}
      %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %12 = nvws.semaphore.buffer %empty, %token_6 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_REINIT:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[NEXT_STAGE]]], [[PTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store {{%.*}}, [[BUF_REINIT]][], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[PTOK]], [[NEXT_STAGE]], [[FPF_NEW]], [[FPE_NEW]]
      %13 = ttng.tmem_store %cst, %12[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %token_6 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_next_iter_user
  tt.func @matmul_tma_acc_with_conditional_next_iter_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
    %empty = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[USE_D:%.*]] = %true, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], [[USE_D]], %true {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // Stage advance + phase-bitset toggle and extraction BEFORE re-acquire EMPTY.
        // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
        // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[PC1]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
        // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
        // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[PHASE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[PHASE_SHIFT:%.*]] = arith.shli [[PHASE_C1]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[FPE_NEW:%.*]] = arith.xori [[FPE]], [[PHASE_SHIFT]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PHASE_SHR:%.*]] = arith.shrui [[FPE_NEW]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PHASE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[PP_OUT:%.*]] = arith.andi [[PHASE_SHR]], [[PHASE_MASK]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[PP_OUT]]] {ttg.partition = array<i32: 1>}
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], {{%.*}}, [[FPE_NEW]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[FTOK]], [[FSTAGE]], [[FPF]], [[FPE]]
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %13 = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      // CHECK: [[FLAG:%.*]] = arith.xori %{{.*}}, %true {ttg.partition = array<i32: 0, 1>} : i1
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF:%.*]]#0, [[FLAG]], [[IF]]#1, [[IF]]#2, [[IF]]#3
      scf.yield %12, %13 : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_next_iter_and_post_loop_read
  // CHECK: [[EMPTY2:%.*]] = nvws.semaphore.create %{{.*}} released = 3
  // CHECK: [[FULL2:%.*]] = nvws.semaphore.create %{{.*}}
  // CHECK: [[FOR2:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK2:%.*]] = %{{.*}}, {{%.*}} = %true, [[FSTAGE2:%.*]] = %{{.*}}, {{%.*}} = %{{.*}}, {{%.*}} = %{{.*}}) -> (!ttg.async.token, i1, i32, i32, i32)
  // CHECK: [[BUF_MMA2:%.*]] = nvws.semaphore.buffer [[EMPTY2]][[[FSTAGE2]]], [[FTOK2]] {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA2]][], {{%.*}}, %true {ttg.partition = array<i32: 1>}
  // CHECK: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]][[[FSTAGE2]], {{%.*}}] {ttg.partition = array<i32: 1>}
  // CHECK: [[BUF_POST2:%.*]] = nvws.semaphore.buffer [[EMPTY2]][[[FOR2]]#2], [[FOR2]]#0 {ttg.partition = array<i32: 1>}
  // CHECK: ttng.tmem_load [[BUF_POST2]][] {ttg.partition = array<i32: 1>}
  tt.func @matmul_tma_acc_with_conditional_next_iter_and_post_loop_read(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %init = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %store = ttng.tmem_store %cst_0, %init[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token, %arg4 = %true) -> (!ttg.async.token, i1)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %arg4, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %12 = scf.if %11 -> (!ttg.async.token) {
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield %token_6 : !ttg.async.token
      } else {
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %13 = arith.xori %11, %true {ttg.partition = array<i32: 0, 1>} : i1
      scf.yield %12, %13 : !ttg.async.token, i1
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
    %post = nvws.semaphore.buffer %empty, %3#0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %loaded, %tok_end = ttng.tmem_load %post[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<64x64xf16, #shared>, %arg2: !tt.tensordesc<64x64xf16, #shared>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[E0:%.*]] = nvws.semaphore.create [[TMEM0:%.*]] released = 3
    // CHECK-DAG: [[F0:%.*]] = nvws.semaphore.create [[TMEM0]]
    // CHECK-DAG: [[E1:%.*]] = nvws.semaphore.create [[TMEM1:%.*]] released = 1
    // CHECK-DAG: [[F1:%.*]] = nvws.semaphore.create [[TMEM1]]
    // CHECK-DAG: [[E2:%.*]] = nvws.semaphore.create [[TMEM2:%.*]] released = 1
    // CHECK-DAG: [[F2:%.*]] = nvws.semaphore.create [[TMEM2]]
    %empty0 = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full0 = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty0 : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %empty1 = nvws.semaphore.create %result_2 released = 1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full1 = nvws.semaphore.create %result_2 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token_4 = nvws.semaphore.acquire %empty1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %2 = nvws.semaphore.buffer %empty1, %token_4 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %3 = ttng.tmem_store %cst_0, %2[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_5 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %empty2 = nvws.semaphore.create %result_5 released = 1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %full2 = nvws.semaphore.create %result_5 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP:%.*]]:12 = scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[DATA0:%.*]] = [[INIT0:%.*]], [[DATA1:%.*]] = [[INIT1:%.*]], [[TOK0:%.*]] = [[TOK0_INIT:%.*]], [[TOK1:%.*]] = [[TOK1_INIT:%.*]], [[S0_S:%.*]] = [[S0_S_INIT:%.*]], [[S0_PF:%.*]] = [[S0_PF_INIT:%.*]], [[S0_PE:%.*]] = [[S0_PE_INIT:%.*]], [[S1_PF:%.*]] = [[S1_PF_INIT:%.*]], [[S1_PE:%.*]] = [[S1_PE_INIT:%.*]], [[S2_S:%.*]] = [[S2_S_INIT:%.*]], [[S2_PE:%.*]] = [[S2_PE_INIT:%.*]], [[S2_PF:%.*]] = [[S2_PF_INIT:%.*]]) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32)
    %5:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %token_4) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: [[DESCLD1:%.*]] = tt.descriptor_load [[ARG1:%.*]][[[IV]], [[C0_LD:%.*]]] {ttg.partition = array<i32: 2>}
      %7 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %8 = ttg.local_alloc %7 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      // CHECK: [[BUF_E0:%.*]] = nvws.semaphore.buffer [[E0]][[[S0_S]]], [[TOK0]] {ttg.partition = array<i32: 1>}
      %10 = nvws.semaphore.buffer %empty0, %arg8 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: [[MMA1:%.*]] = ttng.tc_gen5_mma [[ARG0:%.*]], [[TRANS:%.*]], [[BUF_E0]][], [[FALSE:%.*]], [[TRUE:%.*]] {ttg.partition = array<i32: 1>}
      %11 = ttng.tc_gen5_mma %arg0, %9, %10[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: nvws.semaphore.release [[F0]][[[S0_S]]], [[TOK0]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %full0, %arg8 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase-bitset toggle and extraction BEFORE acquire F0
      // CHECK: [[S0_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S0_PF_SHIFT:%.*]] = arith.shli [[S0_PF_C1]], [[S0_S]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S0_PF_NEW:%.*]] = arith.xori [[S0_PF]], [[S0_PF_SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S0_PF_SHR:%.*]] = arith.shrui [[S0_PF_NEW]], [[S0_S]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S0_PF_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S0_PF_BIT:%.*]] = arith.andi [[S0_PF_SHR]], [[S0_PF_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0_S]], [[S0_PF_BIT]]] {ttg.partition = array<i32: 0>}
      %token_11 = nvws.semaphore.acquire %full0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %12 = nvws.semaphore.buffer %full0, %token_11 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: [[BUF_F0:%.*]] = nvws.semaphore.buffer [[F0]][[[S0_S]]], [[GTOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: [[TLOAD0:%.*]], [[TLTOK0:%.*]] = ttng.tmem_load [[BUF_F0]][] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[E0]][[[S0_S]]], [[GTOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_12, %token_13 = ttng.tmem_load %12[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.semaphore.release %empty0, %token_11 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %13 = "compute_row_max"(%result_12, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = "sub_row_max"(%result_12, %13, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %15 = math.exp2 %14 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %16 = arith.subf %arg7, %13 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.subf %arg7, %13 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = math.exp2 %16 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = math.exp2 %17 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %20 = "tt.reduce"(%15) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %36 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>}: f32
        tt.reduce.return %36 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = arith.mulf %arg6, %19 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addf %21, %20 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.expand_dims %18 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %24 = tt.broadcast %23 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      // CHECK: [[BUF_E1:%.*]] = nvws.semaphore.buffer [[E1]][{{.*}}], [[TOK1]] {ttg.partition = array<i32: 3>}
      %25 = nvws.semaphore.buffer %empty1, %arg9 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: ttng.tmem_load [[BUF_E1]][] {ttg.partition = array<i32: 3>}
      %result_14, %token_15 = ttng.tmem_load %25[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %26 = arith.mulf %result_14, %24 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %27 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %28 = ttg.local_alloc %27 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %29 = arith.truncf %15 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %token_17 = nvws.semaphore.acquire %empty2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %30 = nvws.semaphore.buffer %empty2, %token_17 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // Stage advance, then phase-bitset toggle and extraction BEFORE acquire E2
      // CHECK: [[S2_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S2_NEXT_RAW:%.*]] = arith.addi [[S2_S]], [[S2_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S2_WRAP:%.*]] = arith.cmpi eq, [[S2_NEXT_RAW]], [[S2_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S2_STAGE:%.*]] = arith.select [[S2_WRAP]], [[S2_C0]], [[S2_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S2_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S2_PE_SHIFT:%.*]] = arith.shli [[S2_PE_C1]], [[S2_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S2_PE_NEW:%.*]] = arith.xori [[S2_PE]], [[S2_PE_SHIFT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S2_PE_SHR:%.*]] = arith.shrui [[S2_PE_NEW]], [[S2_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[S2_PE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[S2_PE_BIT:%.*]] = arith.andi [[S2_PE_SHR]], [[S2_PE_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK_E2:%.*]] = nvws.semaphore.acquire [[E2]][[[S2_STAGE]], [[S2_PE_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_E2:%.*]] = nvws.semaphore.buffer [[E2]][[[S2_STAGE]]], [[ATOK_E2]] {ttg.partition = array<i32: 0>}
      // CHECK: [[TSTORE:%.*]] = ttng.tmem_store [[TRUNCF:%.*]], [[BUF_E2]][[[ATOK_E2]]], [[TRUE]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[F2]][[[S2_STAGE]]], [[ATOK_E2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %31 = ttng.tmem_store %29, %30[%token_17], %true {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %full2, %token_17 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %32 = ttng.tmem_store %26, %25[], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.release [[F1]][{{.*}}], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      nvws.semaphore.release %full1, %arg9 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // S1 phase-bitset toggle and extraction without stage advance
      // CHECK: [[S1_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S1_PF_SHIFT:%.*]] = arith.shli [[S1_PF_C1]], [[S1_STAGE:%.*]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_PF_NEW:%.*]] = arith.xori [[S1_PF]], [[S1_PF_SHIFT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S1_PF_SHR:%.*]] = arith.shrui [[S1_PF_NEW]], [[S1_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S1_PF_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S1_PF_BIT:%.*]] = arith.andi [[S1_PF_SHR]], [[S1_PF_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK_F1:%.*]] = nvws.semaphore.acquire [[F1]][[[S1_STAGE]], [[S1_PF_BIT]]] {ttg.partition = array<i32: 1>}
      %token_19 = nvws.semaphore.acquire %full1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.buffer %full1, %token_19 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: [[BUF_F1:%.*]] = nvws.semaphore.buffer [[F1]][{{.*}}], [[ATOK_F1]] {ttg.partition = array<i32: 1>}
      // S2 phase-bitset toggle and extraction before acquire F2
      // CHECK: [[S2_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S2_PF_SHIFT:%.*]] = arith.shli [[S2_PF_C1]], [[S2_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S2_PF_NEW:%.*]] = arith.xori [[S2_PF]], [[S2_PF_SHIFT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S2_PF_SHR:%.*]] = arith.shrui [[S2_PF_NEW]], [[S2_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S2_PF_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S2_PF_BIT:%.*]] = arith.andi [[S2_PF_SHR]], [[S2_PF_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK_F2:%.*]] = nvws.semaphore.acquire [[F2]][[[S2_STAGE]], [[S2_PF_BIT]]] {ttg.partition = array<i32: 1>}
      %token_21 = nvws.semaphore.acquire %full2 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %34 = nvws.semaphore.buffer %full2, %token_21 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: [[BUF_F2:%.*]] = nvws.semaphore.buffer [[F2]][[[S2_STAGE]]], [[ATOK_F2]] {ttg.partition = array<i32: 1>}
      // CHECK: [[MMA2:%.*]] = ttng.tc_gen5_mma [[BUF_F2]], [[ALLOC2:%.*]], [[BUF_F1]][], [[TRUE]], [[TRUE]] {ttg.partition = array<i32: 1>}
      %35 = ttng.tc_gen5_mma %34, %28, %33[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.release [[E2]][[[S2_STAGE]]], [[ATOK_F2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][{{.*}}], [[ATOK_F1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty2, %token_21 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty1, %token_19 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // S0 stage advance
      // CHECK: [[S0_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[S0_NEXT:%.*]] = arith.addi [[S0_S]], [[S0_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S0_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[S0_WRAP:%.*]] = arith.cmpi eq, [[S0_NEXT]], [[S0_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[S0_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[S0_STAGE:%.*]] = arith.select [[S0_WRAP]], [[S0_C0]], [[S0_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // Phase-bitset toggle and extraction before acquire E0
      // CHECK: [[S0_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S0_PE_SHIFT:%.*]] = arith.shli [[S0_PE_C1]], [[S0_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S0_PE_NEW:%.*]] = arith.xori [[S0_PE]], [[S0_PE_SHIFT]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S0_PE_SHR:%.*]] = arith.shrui [[S0_PE_NEW]], [[S0_STAGE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[S0_PE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[S0_PE_BIT:%.*]] = arith.andi [[S0_PE_SHR]], [[S0_PE_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK_E0:%.*]] = nvws.semaphore.acquire [[E0]][[[S0_STAGE]], [[S0_PE_BIT]]] {ttg.partition = array<i32: 1>}
      // Phase-bitset toggle and extraction before acquire E1
      // CHECK: [[S1_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 1 : i32
      // CHECK: [[S1_PE_SHIFT:%.*]] = arith.shli [[S1_PE_C1]], [[S1_STAGE]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[S1_PE_NEW:%.*]] = arith.xori [[S1_PE]], [[S1_PE_SHIFT]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[S1_PE_SHR:%.*]] = arith.shrui [[S1_PE_NEW]], [[S1_STAGE]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[S1_PE_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 1 : i32
      // CHECK: [[S1_PE_BIT:%.*]] = arith.andi [[S1_PE_SHR]], [[S1_PE_MASK]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[ATOK_E1:%.*]] = nvws.semaphore.acquire [[E1]][[[S1_STAGE]], [[S1_PE_BIT]]] {ttg.partition = array<i32: 3>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} [[YIELD_D0:%.*]], [[YIELD_D1:%.*]], [[ATOK_E0]], [[ATOK_E1]], [[S0_STAGE]], [[S0_PF_NEW]], [[S0_PE_NEW]], [[S1_PF_NEW]], [[S1_PE_NEW]], [[S2_STAGE]], [[S2_PE_NEW]], [[S2_PF_NEW]]
      %token_23 = nvws.semaphore.acquire %empty0 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %token_25 = nvws.semaphore.acquire %empty1 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      scf.yield %22, %13, %token_23, %token_25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>, array<i32: 0, 1>, array<i32: 3>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]
    // CHECK: nvws.semaphore.release [[F1]][{{.*}}], [[LOOP]]#3 [#nvws.async_op<tc5mma>]
    // CHECK: nvws.semaphore.release [[F0]][[[LOOP]]#4], [[LOOP]]#2 [#nvws.async_op<none>]
    nvws.semaphore.release %full1, %5#3 [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.semaphore.release %full0, %5#2 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // Phase flip BEFORE acquire F1
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[F1]]
    // CHECK: [[BUF_F1_END:%.*]] = nvws.semaphore.buffer [[F1]][{{.*}}], [[TOK_END]]
    // CHECK: ttng.tmem_load [[BUF_F1_END]][]
    // CHECK: nvws.semaphore.release [[E1]][{{.*}}], [[TOK_END]] [#nvws.async_op<none>]
    %token_7 = nvws.semaphore.acquire %full1 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %full1, %token_7 : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_8, %token_9 = ttng.tmem_load %6[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    nvws.semaphore.release %empty1, %token_7 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%5#0, %result_8, %5#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
    tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}}
    // CHECK: %{{.*}} = nvws.semaphore.acquire [[EMPTY]][{{%.*}}, {{%.*}}]
    // CHECK: %{{.*}} = nvws.semaphore.buffer [[EMPTY]][{{%.*}}], %{{.*}}
    // CHECK: ttng.tmem_store {{%.*}}, %{{.*}}[], {{%.*}}
    %empty = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %token = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %1 = nvws.semaphore.buffer %empty, %token : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:4 = scf.for {{.*}} iter_args([[FTOK:%.*]] = %{{.*}}, [[FSTAGE:%.*]] = %{{.*}}, [[FPF:%.*]] = %{{.*}}, [[FPE:%.*]] = %{{.*}}) -> (!ttg.async.token, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF_MMA:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[FSTAGE]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_MMA]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[IF:%.*]]:4 = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[FULL]][[[FSTAGE]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Phase-bitset toggle and extraction BEFORE acquire FULL
        // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
        // CHECK: [[G_SHIFT:%.*]] = arith.shli [[GC1]], [[FSTAGE]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GP_OUT:%.*]] = arith.xori [[FPF:%.*]], [[G_SHIFT]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[G_SHR:%.*]] = arith.shrui [[GP_OUT]], [[FSTAGE]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[G_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
        // CHECK: [[GP_BIT:%.*]] = arith.andi [[G_SHR]], [[G_MASK]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[FSTAGE]], [[GP_BIT]]] {ttg.partition = array<i32: 0>}
        %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %15 = nvws.semaphore.buffer %full, %token_2 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: [[BUF_LOAD:%.*]] = nvws.semaphore.buffer [[FULL]][[[FSTAGE]]], [[GTOK]] {ttg.partition = array<i32: 0>}
        // CHECK: ttng.tmem_load [[BUF_LOAD]][] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[EMPTY]][[[FSTAGE]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // Stage advance + phase-bitset toggle and extraction BEFORE re-acquire EMPTY.
        // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
        // CHECK: [[NEXT_STAGE_RAW:%.*]] = arith.addi [[FSTAGE]], [[PC1]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
        // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT_STAGE_RAW]], [[DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
        // CHECK: [[NEXT_STAGE:%.*]] = arith.select [[WRAP]], [[ZERO]], [[NEXT_STAGE_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[P_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[P_SHIFT:%.*]] = arith.shli [[P_C1]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PP_OUT:%.*]] = arith.xori [[FPE]], [[P_SHIFT]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[P_SHR:%.*]] = arith.shrui [[PP_OUT]], [[NEXT_STAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[P_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[PP_BIT:%.*]] = arith.andi [[P_SHR]], [[P_MASK]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[NEXT_STAGE]], [[PP_BIT]]] {ttg.partition = array<i32: 1>}
        %token_6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[NEXT_STAGE]], [[GP_OUT]], [[PP_OUT]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[FTOK]], [[FSTAGE]], [[FPF]], [[FPE]]
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      // CHECK: [[BUF_POST:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[IF]]#1], [[IF]]#0 {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store {{%.*}}, [[BUF_POST]][], {{%.*}} {ttg.partition = array<i32: 1>}
      %13 = nvws.semaphore.buffer %empty, %12 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF]]#0, [[IF]]#1, [[IF]]#2, [[IF]]#3
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_tma_persistent_ws_kernel
  tt.func public @matmul_tma_persistent_ws_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg6, %arg8], [%0, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x128xf8E4M3FN, #shared>
    %2 = arith.extsi %arg4 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg7, %arg8], [%2, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x128xf8E4M3FN, #shared>
    %4 = arith.extsi %arg5 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg6, %arg7], [%4, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<128x128xf8E4M3FN, #shared>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg6, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.addi %arg8, %c127_i32 : i32
    %12 = arith.divsi %11, %c128_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %ab_empty = nvws.semaphore.create %15, %16 released = 7 : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %ab_full = nvws.semaphore.create %15, %16 : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[AB_EMPTY:%.*]] = nvws.semaphore.create [[AB_BUF0:%.*]], [[AB_BUF1:%.*]] released = 7
    // CHECK-DAG: [[AB_FULL:%.*]] = nvws.semaphore.create [[AB_BUF2:%.*]], [[AB_BUF3:%.*]]
    // CHECK-DAG: [[ACC_EMPTY:%.*]] = nvws.semaphore.create [[ACC_BUF0:%.*]] released = 3
    // CHECK-DAG: [[ACC_FULL:%.*]] = nvws.semaphore.create [[ACC_BUF1:%.*]]
    %empty_acc = nvws.semaphore.create %result released = 3 : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full_acc = nvws.semaphore.create %result : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER:%.*]]:6 = scf.for [[OUTER_IV:%.*]] = [[OUTER_LB:%.*]] to [[OUTER_UB:%.*]] step [[OUTER_STEP:%.*]] iter_args([[AB_S:%.*]] = [[AB_S_INIT:%.*]], [[AB_PF:%.*]] = [[AB_PF_INIT:%.*]], [[AB_PE:%.*]] = [[AB_PE_INIT:%.*]], [[ACC_S:%.*]] = [[ACC_S_INIT:%.*]], [[ACC_PE:%.*]] = [[ACC_PE_INIT:%.*]], [[ACC_PF:%.*]] = [[ACC_PF_INIT:%.*]]) -> (i32, i32, i32, i32, i32, i32)
    scf.for %arg9 = %6 to %13 step %c148_i32  : i32 {
      %20 = arith.divsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %21 = arith.muli %20, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %22 = arith.subi %8, %21 {ttg.partition = array<i32: 0, 2>} : i32
      %23 = arith.minsi %22, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %24 = arith.remsi %arg9, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %25 = arith.addi %21, %24 {ttg.partition = array<i32: 0, 2>} : i32
      %26 = arith.remsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %27 = arith.divsi %26, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %28 = arith.muli %25, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %29 = arith.muli %27, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      // ACC stage advance: addi/cmpi/select wrapping at depth=2
      // CHECK: [[ACC_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[ACC_S_NEXT:%.*]] = arith.addi [[ACC_S]], [[ACC_C1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[ACC_S_WRAP:%.*]] = arith.cmpi eq, [[ACC_S_NEXT]], [[ACC_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_C0:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[ASTAGE0:%.*]] = arith.select [[ACC_S_WRAP]], [[ACC_C0]], [[ACC_S_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // Phase flip BEFORE acquire ACC_EMPTY (partition 0)
      // CHECK: [[ACC_PE_C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ACC_PE_SHIFT0:%.*]] = arith.shli [[ACC_PE_C1]], [[ASTAGE0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0:%.*]] = arith.xori [[ACC_PE]], [[ACC_PE_SHIFT0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0_SHR:%.*]] = arith.shrui [[APHASE0]], [[ASTAGE0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[APHASE0_BIT:%.*]] = arith.andi [[APHASE0_SHR]], [[APHASE0_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK0:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE0]], [[APHASE0_BIT]]] {ttg.partition = array<i32: 0>}
      %token = nvws.semaphore.acquire %empty_acc {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %30 = nvws.semaphore.buffer %empty_acc, %token {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: [[BUF_ACC:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ASTAGE0]]], [[ATOK0]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}}, [[BUF_ACC]][], {{.*}} {ttg.partition = array<i32: 0>}
      %31 = ttng.tmem_store %cst, %30[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.semaphore.release [[ACC_FULL]][[[ASTAGE0]]], [[ATOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %full_acc, %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE acquire ACC_FULL (partition 1)
      // CHECK: [[ACC_C1_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[ACC_S_NEXT1:%.*]] = arith.addi [[ASTAGE0]], [[ACC_C1_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_DEPTH_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 2 : i32
      // CHECK: [[ACC_S_WRAP1:%.*]] = arith.cmpi eq, [[ACC_S_NEXT1]], [[ACC_DEPTH_NEXT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_C0_NEXT:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[ASTAGE1:%.*]] = arith.select [[ACC_S_WRAP1]], [[ACC_C0_NEXT]], [[ACC_S_NEXT1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[ACC_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[ACC_PF_SHIFT0:%.*]] = arith.shli [[ACC_PF_C1]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1:%.*]] = arith.xori [[ACC_PF]], [[ACC_PF_SHIFT0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1_SHR:%.*]] = arith.shrui [[APHASE1]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[APHASE1_BIT:%.*]] = arith.andi [[APHASE1_SHR]], [[APHASE1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK1:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE1]], [[APHASE1_BIT]]] {ttg.partition = array<i32: 1>}
      %token_1 = nvws.semaphore.acquire %full_acc {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[INNER:%.*]]:4 = scf.for [[INNER_IV:%.*]] = [[INNER_LB:%.*]] to [[INNER_UB:%.*]] step [[INNER_STEP:%.*]] iter_args({{%.*}} = [[INNER_USED:%.*]], [[AB_S_I:%.*]] = [[AB_S]], [[AB_PF_I:%.*]] = [[AB_PF]], [[AB_PE_I:%.*]] = [[AB_PE]]) -> (i1, i32, i32, i32)
      %32 = scf.for %arg10 = %c0_i32 to %12 step %c1_i32 iter_args(%arg11 = %false) -> (i1)  : i32 {
        %36 = arith.muli %arg10, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // AB stage advance + phase-bitset toggle and extraction BEFORE acquire AB_EMPTY
        // CHECK: [[AB_C1:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 1 : i32
        // CHECK: [[AB_S_NEXT:%.*]] = arith.addi [[AB_S_I]], [[AB_C1]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_DEPTH:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 3 : i32
        // CHECK: [[AB_S_WRAP:%.*]] = arith.cmpi eq, [[AB_S_NEXT]], [[AB_DEPTH]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_C0:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} 0 : i32
        // CHECK: [[ABSTAGE:%.*]] = arith.select [[AB_S_WRAP]], [[AB_C0]], [[AB_S_NEXT]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
        // CHECK: [[AB_PE_C1:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 1 : i32
        // CHECK: [[AB_PE_SHIFT:%.*]] = arith.shli [[AB_PE_C1]], [[ABSTAGE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[AB_PE_NEW:%.*]] = arith.xori [[AB_PE_I]], [[AB_PE_SHIFT]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[AB_PE_SHR:%.*]] = arith.shrui [[AB_PE_NEW]], [[ABSTAGE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[AB_PE_MASK:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 1 : i32
        // CHECK: [[AB_PE_BIT:%.*]] = arith.andi [[AB_PE_SHR]], [[AB_PE_MASK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[ABTOK_P:%.*]] = nvws.semaphore.acquire [[AB_EMPTY]][[[ABSTAGE]], [[AB_PE_BIT]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        %token_9 = nvws.semaphore.acquire %ab_empty {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[BUFS_AB:%.*]]:2 = nvws.semaphore.buffer [[AB_EMPTY]][[[ABSTAGE]]], [[ABTOK_P]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        %buffers_8:2 = nvws.semaphore.buffer %ab_empty, %token_9 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.descriptor_load {{.*}} [[BUFS_AB]]#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %1[%28, %36] 16384 %buffers_8#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.descriptor_load {{.*}} [[BUFS_AB]]#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.descriptor_load %3[%29, %36] 16384 %buffers_8#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[AB_FULL]][[[ABSTAGE]]], [[ABTOK_P]] [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.semaphore.release %ab_full, %token_9 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token

        // Phase-bitset toggle and extraction BEFORE acquire AB_FULL
        // CHECK: [[AB_PF_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[AB_PF_SHIFT:%.*]] = arith.shli [[AB_PF_C1]], [[ABSTAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[AB_PF_NEW:%.*]] = arith.xori [[AB_PF_I]], [[AB_PF_SHIFT]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[AB_PF_SHR:%.*]] = arith.shrui [[AB_PF_NEW]], [[ABSTAGE]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[AB_PF_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[AB_PF_BIT:%.*]] = arith.andi [[AB_PF_SHR]], [[AB_PF_MASK]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[ABTOK_C:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[ABSTAGE]], [[AB_PF_BIT]]] {ttg.partition = array<i32: 1>}
        %token_11 = nvws.semaphore.acquire %ab_full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[BUFS_ABF:%.*]]:2 = nvws.semaphore.buffer [[AB_FULL]][[[ABSTAGE]]], [[ABTOK_C]] {ttg.partition = array<i32: 1>}
        %buffers_10:2 = nvws.semaphore.buffer %ab_full, %token_11 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        %37 = ttg.memdesc_trans %buffers_10#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>
        // CHECK: [[BUF_ACCF_INNER:%.*]] = nvws.semaphore.buffer [[ACC_FULL]][[[ASTAGE1]]], [[ATOK1]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
        %38 = nvws.semaphore.buffer %full_acc, %token_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tc_gen5_mma [[BUFS_ABF]]#0, {{.*}}, [[BUF_ACCF_INNER]][], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>}
        %39 = ttng.tc_gen5_mma %buffers_10#0, %37, %38[], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: nvws.semaphore.release [[AB_EMPTY]][[[ABSTAGE]]], [[ABTOK_C]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.semaphore.release %ab_empty, %token_11 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} {{%.*}}, [[ABSTAGE]], [[AB_PF_NEW]], [[AB_PE_NEW]]
        scf.yield %true : i1
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_acc, %token_1 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %token_3 = nvws.semaphore.acquire %empty_acc {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.buffer %empty_acc, %token_3 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_4, %token_5 = ttng.tmem_load %33[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %full_acc, %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase flip BEFORE acquire ACC_EMPTY
      // CHECK: [[ACC_PE_SHIFT2:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[ACC_PE_SHIFT2_V:%.*]] = arith.shli [[ACC_PE_SHIFT2]], [[ASTAGE1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2:%.*]] = arith.xori [[APHASE0]], [[ACC_PE_SHIFT2_V]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2_SHR:%.*]] = arith.shrui [[APHASE2]], [[ASTAGE1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[APHASE2_BIT:%.*]] = arith.andi [[APHASE2_SHR]], [[APHASE2_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK2:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE1]], [[APHASE2_BIT]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_ACCE2:%.*]] = nvws.semaphore.buffer [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK2]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_load [[BUF_ACCE2]][] {ttg.partition = array<i32: 0>}
      %token_7 = nvws.semaphore.acquire %full_acc {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[ACC_FULL]][[[ASTAGE1]]], [[ATOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Phase flip BEFORE acquire ACC_FULL
      // CHECK: [[ACC_PF_SHIFT2:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[ACC_PF_SHIFT2_V:%.*]] = arith.shli [[ACC_PF_SHIFT2]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3:%.*]] = arith.xori [[APHASE1]], [[ACC_PF_SHIFT2_V]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3_SHR:%.*]] = arith.shrui [[APHASE3]], [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[APHASE3_BIT:%.*]] = arith.andi [[APHASE3_SHR]], [[APHASE3_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK3:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE1]], [[APHASE3_BIT]]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][[[ASTAGE1]]], [[ATOK3]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_acc, %token_7 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %34 = tt.fp_to_fp %result_4, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %35 = ttg.convert_layout %34 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %5[%28, %29], %35 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, tensor<128x128xf8E4M3FN, #blocked1>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[INNER]]#1, [[INNER]]#2, [[INNER]]#3, [[ASTAGE1]], [[APHASE2]], [[APHASE3]]
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>, array<i32: 2>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @for_loop_control_operand_ppg
  tt.func @for_loop_control_operand_ppg(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %semBuf = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create {{.*}} released = 1
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create {{.*}}
    // CHECK: [[S0:%.*]] = arith.constant 0 : i32
    // CHECK: [[PE_INIT:%.*]] = arith.constant -2 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant -1 : i32
    // Pre-loop: phase-bitset toggles and extraction BEFORE acquire EMPTY
    // CHECK: [[PF1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_SHIFT:%.*]] = arith.shli [[PF1_C1]], [[S0]] : i32
    // CHECK: [[PF1_OUT:%.*]] = arith.xori [[PE_INIT]], [[PF1_SHIFT]] : i32
    // CHECK: [[PF1_SHR:%.*]] = arith.shrui [[PF1_OUT]], [[S0]] : i32
    // CHECK: [[PF1_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[PF1_BIT:%.*]] = arith.andi [[PF1_SHR]], [[PF1_MASK]] : i32
    // CHECK: [[PE1_C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[PE1_SHIFT:%.*]] = arith.shli [[PE1_C1]], [[S0]] : i32
    // CHECK: [[P0_OUT:%.*]] = arith.xori [[PE_INIT]], [[PE1_SHIFT]] : i32
    // CHECK: [[P0_SHR:%.*]] = arith.shrui [[P0_OUT]], [[S0]] : i32
    // CHECK: [[P0_MASK:%.*]] = arith.constant 1 : i32
    // CHECK: [[P0_BIT:%.*]] = arith.andi [[P0_SHR]], [[P0_MASK]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0]], [[P0_BIT]]]
    %empty = nvws.semaphore.create %semBuf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %semBuf : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %tok = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[FOR0:%.*]]:3 = scf.for {{.*}} iter_args([[F0TOK:%.*]] = [[TOK0]], [[F0PF:%.*]] = [[PF_INIT]], [[F0PE:%.*]] = [[P0_OUT]]) -> (!ttg.async.token, i32, i32)
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
      %ptrub = tt.addptr %ptr0, %iv0 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %ub1 = tt.load %ptrub {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %lb1 = "lb1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %step1 = "step1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      // CHECK: scf.for {{.*}} : i32 {
      %tok5 = scf.for %iv = %lb1 to %ub1 step %step1 iter_args(%tok2 = %tok1) -> (!ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[BUF_INNER:%.*]] = nvws.semaphore.buffer [[EMPTY]][[[S0]]], [[F0TOK]] {ttg.partition = array<i32: 2>}
        %buf = nvws.semaphore.buffer %empty, %tok2 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[BUF_INNER]], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>}
        ttng.tc_gen5_mma %sA, %sB, %buf, %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %tok2 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2>}
      // CHECK: nvws.semaphore.release [[FULL]][[[S0]]], [[F0TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>}
      nvws.semaphore.release %full, %tok5 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase-bitset toggle and extraction BEFORE acquire FULL
      // CHECK: [[GC1:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[GC1_SHIFT:%.*]] = arith.shli [[GC1]], [[S0]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P1_OUT:%.*]] = arith.xori [[F0PF:%.*]], [[GC1_SHIFT]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P1_SHR:%.*]] = arith.shrui [[P1_OUT]], [[S0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[P1_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[P1_BIT:%.*]] = arith.andi [[P1_SHR]], [[P1_MASK]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[S0]], [[P1_BIT]]] {ttg.partition = array<i32: 1>}
      %token_2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]][[[S0]]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // Phase-bitset toggle and extraction BEFORE re-acquire EMPTY
      // CHECK: [[PC1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[PC1_SHIFT:%.*]] = arith.shli [[PC1]], [[S0]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[P2_OUT:%.*]] = arith.xori [[F0PE:%.*]], [[PC1_SHIFT]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[P2_SHR:%.*]] = arith.shrui [[P2_OUT]], [[S0]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[P2_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[P2_BIT:%.*]] = arith.andi [[P2_SHR]], [[P2_MASK]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0]], [[P2_BIT]]] {ttg.partition = array<i32: 2>}
      %tok6 = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[TOK2]], [[P1_OUT]], [[P2_OUT]]
      scf.yield {ttg.partition = array<i32: 1, 2>} %tok6 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 0, 1>, array<i32: 2>]}
    // CHECK: nvws.semaphore.release [[FULL]][[[S0]]], [[FOR0]]#0 [#nvws.async_op<tc5mma>]
    // Post-loop: phase-bitset toggles and extraction BEFORE acquire FULL
    // CHECK: [[POST_C1A:%.*]] = arith.constant 1 : i32
    // CHECK: [[POST_SHIFTA:%.*]] = arith.shli [[POST_C1A]], [[S0]] : i32
    // CHECK: [[POST_OUT_A:%.*]] = arith.xori [[FOR0]]#1, [[POST_SHIFTA]] : i32
    // CHECK: [[POST_SHRA:%.*]] = arith.shrui [[POST_OUT_A]], [[S0]] : i32
    // CHECK: [[POST_MASKA:%.*]] = arith.constant 1 : i32
    // CHECK: [[POST_BITA:%.*]] = arith.andi [[POST_SHRA]], [[POST_MASKA]] : i32
    // CHECK: [[POST_C1B:%.*]] = arith.constant 1 : i32
    // CHECK: [[POST_SHIFTB:%.*]] = arith.shli [[POST_C1B]], [[S0]] : i32
    // CHECK: [[POST_OUT_B:%.*]] = arith.xori [[PF_INIT]], [[POST_SHIFTB]] : i32
    // CHECK: [[POST_SHRB:%.*]] = arith.shrui [[POST_OUT_B]], [[S0]] : i32
    // CHECK: [[POST_MASKB:%.*]] = arith.constant 1 : i32
    // CHECK: [[P_END:%.*]] = arith.andi [[POST_SHRB]], [[POST_MASKB]] : i32
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[FULL]][[[S0]], [[P_END]]]
    // CHECK: nvws.semaphore.release [[EMPTY]][[[S0]]], [[TOK_END]] [#nvws.async_op<none>]
    nvws.semaphore.release %full, %tok0 [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %token_end = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %empty, %token_end [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @loop_carried_authored_stage_buffer_metadata
  tt.func @loop_carried_authored_stage_buffer_metadata(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} released = 3
    // CHECK: [[FOR:%.*]]:5 = scf.for
    // CHECK: nvws.semaphore.buffer [[SEM]][{{%.*}}], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
    // CHECK: nvws.semaphore.buffer [[SEM]][{{%.*}}], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 0, 1, 2>
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok0 = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %tok1 = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %tok2:2 = scf.for %i = %lb to %ub step %step iter_args(%tok_a = %tok0, %tok_b = %tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      %slot = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %view0 = nvws.semaphore.buffer %sem[%slot], %tok_a {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem[%slot], %tok_a [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %slot_1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %view1 = nvws.semaphore.buffer %sem[%slot_1], %tok_b {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem[%slot_1], %tok_b [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %next_a = nvws.semaphore.acquire %sem {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %next_b = nvws.semaphore.acquire %sem {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1, 2>} %next_a, %next_b : !ttg.async.token, !ttg.async.token
    } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
//--- circular.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  tt.func @circular_model_descriptor_load_mma(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]]
    // CHECK: [[LOOP:%.*]]:3 = scf.for {{.*}} iter_args([[STAGE_IN:%[^ ]+]] = {{%[^,]+}}, {{.*}}) -> (i32, i32, i32)
    %empty = nvws.semaphore.create %base released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_C1_STEP:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 1 : i32
      // CHECK: [[K_C1_NEXT:%.*]] = arith.addi [[STAGE_IN]], [[K_C1_STEP]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_C1_DEPTH:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 2 : i32
      // CHECK: [[K_C1_WRAP:%.*]] = arith.cmpi eq, [[K_C1_NEXT]], [[K_C1_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_C1_ZERO:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 0 : i32
      // CHECK: [[K_STAGE_C1:%.*]] = arith.select [[K_C1_WRAP]], [[K_C1_ZERO]], [[K_C1_NEXT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE_C1]], [[K_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE_C1]]], [[K_EMPTY_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE_C1]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_C4_STEP:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 1 : i32
      // CHECK: [[V_C4_NEXT:%.*]] = arith.addi [[K_STAGE_C1]], [[V_C4_STEP]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_C4_DEPTH:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 2 : i32
      // CHECK: [[V_C4_WRAP:%.*]] = arith.cmpi eq, [[V_C4_NEXT]], [[V_C4_DEPTH]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_C4_ZERO:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 0 : i32
      // CHECK: [[V_STAGE_C4:%.*]] = arith.select [[V_C4_WRAP]], [[V_C4_ZERO]], [[V_C4_NEXT]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_C1_NEXT:%.*]] = arith.addi [[K_STAGE_C1]], [[K_C1_STEP]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_STAGE_C1:%.*]] = arith.select {{%.*}}, [[K_C1_ZERO]], [[V_C1_NEXT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE_C4]], [[V_PHASE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE_C4]]], [[V_EMPTY_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE_C4]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE_C1]], [[K_ACQ_M1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_C1_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_C1_ZERO]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_C1_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE_C1]], [[K_BUF_M1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1_k], %tk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1_k], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v_use = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE_C4]], [[V_FULL_PHASE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE_C4]]], [[V_FULL_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE_C4]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z_v_use] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z_v_use], %tv_use {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z_v_use], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 3>} [[V_STAGE_C4]], {{%.*}}, {{%.*}} : i32, i32, i32
    } {tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_2
  tt.func @circular_tutorial_1_1_to_2_2(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]]
    %empty = nvws.semaphore.create %base released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z1], %tv_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z1], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_4
  tt.func @circular_tutorial_1_2_to_3_4(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]]
    %empty = nvws.semaphore.create %base released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z] {ttg.partition = array<i32: 4>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z], %tv_use {ttg.partition = array<i32: 4>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 3, 4>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_3
  tt.func @circular_tutorial_1_1_to_2_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]]
    %empty = nvws.semaphore.create %base released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z1], %tv_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z1], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 2, 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_3
  tt.func @circular_tutorial_1_2_to_3_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] released = 3
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]]
    %empty = nvws.semaphore.create %base released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z], %tv_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }
}

//--- from-insert.mlir

// Assign concrete stages and phases to protocols emitted from managed buffers.

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @fused_alias_depth_two
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    // Both member allocs collapse onto one fused depth-2 backing allocation;
    // every semaphore lists both (identical) member views as its buffers.
    // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] released = 3 {pending_count = 1 : i32}
    // CHECK: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // CHECK: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    // CHECK: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] {pending_count = 1 : i32}
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>

    // The loop-close release partition (2) differs from the first-acquire
    // partition (4), so no acquire token is threaded through iter_args before
    // stage/phase assignment. ASP threads the slot cursor plus one phase word
    // per acquirer.
    // CHECK: scf.for {{.*}} iter_args([[CURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: acquire ENTRY, store through view #0, release FULL0.
      // CHECK: [[SLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: arith.shli {{%.*}}, [[SLOT0]] {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[PHN_W0:%.*]] = arith.xori [[PH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[W0_BUF:%.*]]:2 = nvws.semaphore.buffer [[ENTRY]][[[SLOT0]]], [[W0_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttg.local_store {{%.*}}, [[W0_BUF]]#0 {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[FULL0]][[[SLOT0]]], [[W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 0 read: acquire FULL0, load view #0, release EMPTY1 at the
      // successor slot (SLOT0 + 1).
      // CHECK: [[PHN_R0:%.*]] = arith.xori [[PH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[SLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[R0_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL0]][[[SLOT0]]], [[R0_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[R0_BUF]]#0 {ttg.partition = array<i32: 2>}
      // CHECK: [[TO_M1_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_M1_REM:%.*]] = arith.remsi [[TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0 = ttg.local_load %m0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      // Member 1 write: acquire EMPTY1, store through view #1, release FULL1.
      // CHECK: [[NEXT_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[SLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[NEXT_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: arith.shli {{%.*}}, [[SLOT1]] {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[PHN_W1:%.*]] = arith.xori [[PH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[W1_BUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY1]][[[SLOT1]]], [[W1_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttg.local_store {{%.*}}, [[W1_BUF]]#1 {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[FULL1]][[[SLOT1]]], [[W1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Member 1 read: acquire FULL1, load view #1, close the loop by
      // releasing ENTRY at the successor slot (SLOT1 + 1) mod 2.
      // CHECK: [[PHN_R1:%.*]] = arith.xori [[PH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[SLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[R1_BUF:%.*]]:2 = nvws.semaphore.buffer [[FULL1]][[[SLOT1]]], [[R1_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[R1_BUF]]#1 {ttg.partition = array<i32: 2>}
      // CHECK: [[TO_M0_RAW:%.*]] = arith.addi [[SLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_M0_REM:%.*]] = arith.remsi [[TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[ENTRY]][[[TO_M0]]], [[R1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK: scf.yield {ttg.partition = array<i32: 2, 4>} [[SLOT1]], [[PHN_R0]], [[PHN_R1]], [[PHN_W0]], [[PHN_W1]] : i32, i32, i32, i32, i32
      %r1 = ttg.local_load %m1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Planner-authored aliases may be different views of one staged backing.
  // Here the smaller member covers the prefix of the larger member.  The
  // read-to-next-write handoff must still target the following physical slot.
  // CHECK-LABEL: @fused_partial_alias_depth_three
  tt.func @fused_partial_alias_depth_three(%lb: i32, %ub: i32, %step: i32) {
    // Entry stages 0 and 2 are acquired before their first release; stage 1
    // is released before its first acquire, so the bootstrap mask is 0b101.
    // CHECK: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // CHECK: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
    // CHECK: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] released = 5 {pending_count = 1 : i32}
    // CHECK: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // CHECK: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    // CHECK: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] {pending_count = 1 : i32}
    %large = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %small = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %small_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %large_value = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #blocked64>

    // CHECK: scf.for {{.*}} iter_args([[PCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[PPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Small-member write: acquire PENTRY, store through view #1 (the small
      // member), release PFULL0.
      // CHECK: [[PSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[PPHN_W0:%.*]] = arith.xori [[PPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[PW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PENTRY]][[[PSLOT0]]], [[PW0_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttg.local_store {{%.*}}, [[PW0_BUF]]#1 {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[PFULL0]][[[PSLOT0]]], [[PW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %small_value, %small {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Small-member read: acquire PFULL0, load view #1, then hand off to the
      // large write at the following physical slot (PSLOT0 + 1) mod 3.
      // CHECK: [[PPHN_R0:%.*]] = arith.xori [[PPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[PR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL0]][[[PSLOT0]]], [[PR0_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[PR0_BUF]]#1 {ttg.partition = array<i32: 2>}
      // CHECK: [[TO_LARGE_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_LARGE_REM:%.*]] = arith.remsi [[TO_LARGE_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TO_LARGE_SLOT:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_LARGE_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE_SLOT]]], [[PR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %small_read = ttg.local_load %small {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_small"(%small_read) {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked64>) -> ()
      // Large-member write: acquire PHANDOFF at the successor slot, store
      // through view #0 (the large member), release PFULL1.
      // CHECK: [[PSLOT1_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[PSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[PSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[PPHN_W1:%.*]] = arith.xori [[PPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[PW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PHANDOFF]][[[PSLOT1]]], [[PW1_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttg.local_store {{%.*}}, [[PW1_BUF]]#0 {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[PFULL1]][[[PSLOT1]]], [[PW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked64> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // Large-member read: acquire PFULL1, load view #0, close the loop by
      // releasing PENTRY at the reader's own slot (constant 0 / PSLOT1: the
      // slot the next small write reaches two iterations later).
      // CHECK: [[PPHN_R1:%.*]] = arith.xori [[PPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PR1_TOK:%.*]] = nvws.semaphore.acquire [[PFULL1]][[[PSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[PR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[PFULL1]][[[PSLOT1]]], [[PR1_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[PR1_BUF]]#0 {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[PENTRY]][[[PSLOT1]]], [[PR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK: scf.yield {ttg.partition = array<i32: 2, 4>} [[PSLOT1]], [[PPHN_R0]], [[PPHN_R1]], [[PPHN_W0]], [[PPHN_W1]] : i32, i32, i32, i32, i32
      %large_read = ttg.local_load %large {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked64>
      "consume_large"(%large_read) {ttg.partition = array<i32: 2>} : (tensor<256x64xf16, #blocked64>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // CHECK-LABEL: @tmem_fused_alias_depth_two
  tt.func @tmem_fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>

    // The fused tmem allocation and its semaphores are hoisted to function
    // scope, ahead of the loop that contains the source tmem_allocs.
    // CHECK: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32}
    // CHECK: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] released = 3 {pending_count = 1 : i32}
    // CHECK: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // CHECK: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // CHECK: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] {pending_count = 1 : i32}
    // CHECK: scf.for {{.*}} iter_args([[TCURSOR:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_R1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W0:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}}, [[TPH_W1:%[-A-Za-z0-9_.$#]+]] = %{{[-A-Za-z0-9_.$#]+}})
    scf.for %iv = %lb to %ub step %step : i32 {
      // Member 0 write: the value-carrying tmem_alloc becomes a tmem_store
      // through view #0 with no token bracket.
      // CHECK: [[TSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[TPHN_W0:%.*]] = arith.xori [[TPH_W0]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[TW0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TENTRY]][[[TSLOT0]]], [[TW0_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttng.tmem_store {{%.*}}, [[TW0_BUF]]#0, {{%.*}} {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[TFULL0]][[[TSLOT0]]], [[TW0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m0 = ttng.tmem_alloc %v0 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 0 read: acquire TFULL0, load view #0 with an empty token
      // bracket, release TEMPTY1 at the successor slot.
      // CHECK: [[TPHN_R0:%.*]] = arith.xori [[TPH_R0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TSLOT0]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[TR0_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL0]][[[TSLOT0]]], [[TR0_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.tmem_load [[TR0_BUF]]#0[] {ttg.partition = array<i32: 2>}
      // CHECK: [[T_TO_M1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[T_TO_M1_REM:%.*]] = arith.remsi [[T_TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[T_TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      %r0, %t0 = ttng.tmem_load %m0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      // Member 1 write: acquire TEMPTY1, store through view #1, release
      // TFULL1.
      // CHECK: [[TSLOT1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[TSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
      // CHECK: [[TPHN_W1:%.*]] = arith.xori [[TPH_W1]], {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 4>}
      // CHECK: [[TW1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TEMPTY1]][[[TSLOT1]]], [[TW1_TOK]] {ttg.partition = array<i32: 4>}
      // CHECK: ttng.tmem_store {{%.*}}, [[TW1_BUF]]#1, {{%.*}} {ttg.partition = array<i32: 4>}
      // CHECK: nvws.semaphore.release [[TFULL1]][[[TSLOT1]]], [[TW1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>}
      %m1 = ttng.tmem_alloc %v1 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // Member 1 read: acquire TFULL1, load view #1, close the loop by
      // releasing TENTRY at the successor slot (TSLOT1 + 1) mod 2.
      // CHECK: [[TPHN_R1:%.*]] = arith.xori [[TPH_R1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TSLOT1]], {{%.*}}] {ttg.partition = array<i32: 2>}
      // CHECK: [[TR1_BUF:%.*]]:2 = nvws.semaphore.buffer [[TFULL1]][[[TSLOT1]]], [[TR1_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.tmem_load [[TR1_BUF]]#1[] {ttg.partition = array<i32: 2>}
      // CHECK: [[T_TO_M0_RAW:%.*]] = arith.addi [[TSLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[T_TO_M0_REM:%.*]] = arith.remsi [[T_TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[T_TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK: scf.yield {ttg.partition = array<i32: 2, 4>} [[TSLOT1]], [[TPHN_R0]], [[TPHN_R1]], [[TPHN_W0]], [[TPHN_W1]] : i32, i32, i32, i32, i32
      %r1, %t1 = ttng.tmem_load %m1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // A: both fresh-write epochs are inside one loop.
  // CHECK-LABEL: @case_a
  tt.func @case_a(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // CHECK: [[A_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // CHECK: [[A_ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[A_NEXT:%.*]] = nvws.semaphore.create [[A_BASE]] released = 6 {pending_count = 1 : i32}
    // CHECK: [[A_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // CHECK: [[A_INITIAL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 1 : i32
    // CHECK: [[A_INITIAL_NEXT_RAW:%.*]] = arith.addi [[A_INITIAL_CURRENT_STAGE]], [[A_INITIAL_ONE]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // CHECK: [[A_INITIAL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 3 : i32
    // CHECK: [[A_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_INITIAL_NEXT_RAW]], [[A_INITIAL_DEPTH]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // CHECK: [[A_INITIAL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} 0 : i32
    // CHECK: [[A_INITIAL_NEXT_STAGE:%.*]] = arith.select [[A_INITIAL_NEEDS_WRAP]], [[A_INITIAL_ZERO]], [[A_INITIAL_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 10 : i32} : i32
    // CHECK: [[A_INIT_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_INITIAL_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 10 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for {{.*}} iter_args([[A_W0_TOK:%.*]] = [[A_INIT_TOK]], [[A_CURRENT_STAGE:%.*]] = [[A_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[A_W0_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{%.*}}, [[A_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[A_NEXT]][[[A_CURRENT_STAGE]]], [[A_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%carry], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[A_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[A_NEXT_RAW:%.*]] = arith.addi [[A_CURRENT_STAGE]], [[A_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[A_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[A_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[A_NEXT_RAW]], [[A_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[A_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[A_NEXT_STAGE:%.*]] = arith.select [[A_NEEDS_WRAP]], [[A_ZERO]], [[A_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[A_W1_TOK:%.*]] = nvws.semaphore.acquire [[A_NEXT]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[A_W1_BUF:%.*]] = nvws.semaphore.buffer [[A_NEXT]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[A_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%w0], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_ENTRY]][[[A_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[A_READ_BUF:%.*]] = nvws.semaphore.buffer [[A_ENTRY]][[[A_NEXT_STAGE]]], [[A_READ_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: {{%.*}}, {{%.*}} = ttng.tmem_load [[A_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_a"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 10 : i32}
    tt.return
  }

  // B: the first write precedes the loop, so the default all/none masks stay.
  // CHECK-LABEL: @case_b
  tt.func @case_b(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // CHECK: [[B_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // CHECK: [[B_ENTRY:%.*]] = nvws.semaphore.create [[B_BASE]] released = 7 {pending_count = 1 : i32}
    // CHECK: [[B_FULL:%.*]] = nvws.semaphore.create [[B_BASE]] {pending_count = 1 : i32}
    // CHECK: [[B_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // CHECK: [[B_INITIAL_ONE:%.*]] = arith.constant 1 : i32
    // CHECK: [[B_INITIAL_NEXT_RAW:%.*]] = arith.addi [[B_INITIAL_CURRENT_STAGE]], [[B_INITIAL_ONE]] : i32
    // CHECK: [[B_INITIAL_DEPTH:%.*]] = arith.constant 3 : i32
    // CHECK: [[B_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_INITIAL_NEXT_RAW]], [[B_INITIAL_DEPTH]] : i32
    // CHECK: [[B_INITIAL_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK: [[B_INITIAL_NEXT_STAGE:%.*]] = arith.select [[B_INITIAL_NEEDS_WRAP]], [[B_INITIAL_ZERO]], [[B_INITIAL_NEXT_RAW]] : i32
    // CHECK: [[B_INIT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]], {{%.*}}]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[B_W0_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_INITIAL_NEXT_STAGE]]], [[B_INIT_TOK]]
    // CHECK: ttng.tmem_store {{%.*}}, [[B_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
    %w0 = ttng.tmem_store %zero, %acc[%tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for {{.*}} iter_args([[B_MMA_TOK:%.*]] = [[B_INIT_TOK]], [[B_CURRENT_STAGE:%.*]] = [[B_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%carry = %w0) -> (!ttg.async.token) : i32 {
      // CHECK: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[B_MMA_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[B_READ_BUF:%.*]] = nvws.semaphore.buffer [[B_FULL]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: {{%.*}}, {{%.*}} = ttng.tmem_load [[B_READ_BUF]][] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[B_ENTRY]][[[B_CURRENT_STAGE]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_b"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: [[B_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[B_NEXT_RAW:%.*]] = arith.addi [[B_CURRENT_STAGE]], [[B_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[B_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[B_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[B_NEXT_RAW]], [[B_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[B_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[B_NEXT_STAGE:%.*]] = arith.select [[B_NEEDS_WRAP]], [[B_ZERO]], [[B_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[B_NEXT_TOK:%.*]] = nvws.semaphore.acquire [[B_ENTRY]][[[B_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 11 : i32}
    tt.return
  }

  // C: the loop-exit relay reserves the successor slot for the next W0.
  // CHECK-LABEL: @case_c
  tt.func @case_c(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // CHECK: [[C_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // CHECK: [[C_ENTRY:%.*]] = nvws.semaphore.create [[C_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[C_FULL:%.*]] = nvws.semaphore.create [[C_BASE]] {pending_count = 1 : i32}
    // CHECK: [[C_FREE:%.*]] = nvws.semaphore.create [[C_BASE]] released = 6 {pending_count = 1 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for {{.*}} iter_args([[C_OUTER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = {{%[-A-Za-z0-9_.$#]+}},
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[C_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[C_OUTER_NEXT_RAW:%.*]] = arith.addi [[C_OUTER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[C_OUTER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_OUTER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[C_NEXT_OF_OUTER_STAGE:%.*]] = arith.select [[C_OUTER_NEEDS_WRAP]], [[C_ZERO]], [[C_OUTER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_W0_TOK:%.*]] = nvws.semaphore.acquire [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[C_W0_BUF:%.*]] = nvws.semaphore.buffer [[C_ENTRY]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{%.*}}, [[C_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_OUTER_STAGE]]], [[C_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[C_INNER_LOOP:%.*]]:3 = scf.for {{.*}} iter_args([[C_INNER_CURRENT_STAGE:%[-A-Za-z0-9_.$#]+]] = [[C_NEXT_OF_OUTER_STAGE]],
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // CHECK: [[C_INNER_NEXT_RAW:%.*]] = arith.addi [[C_INNER_CURRENT_STAGE]], [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[C_INNER_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[C_INNER_NEXT_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[C_NEXT_OF_INNER_CURRENT_STAGE:%.*]] = arith.select [[C_INNER_NEEDS_WRAP]], [[C_ZERO]], [[C_INNER_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
        // CHECK: [[C_W1_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // CHECK: [[C_W1_BUF:%.*]] = nvws.semaphore.buffer [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] {ttg.partition = array<i32: 1>}
        // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[C_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // CHECK: [[C_READ_BUF:%.*]] = nvws.semaphore.buffer [[C_FULL]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] {ttg.partition = array<i32: 0>}
        // CHECK: {{%.*}}, {{%.*}} = ttng.tmem_load [[C_READ_BUF]][] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[C_FREE]][[[C_NEXT_OF_INNER_CURRENT_STAGE]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        %value, %read = ttng.tmem_load %acc[%w1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_c"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // The accessless relay targets next(inner-stage) on both sides.
      // CHECK: [[C_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_NEXT_OF_INNER_RAW]], [[C_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[C_REL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_FREE]][[[C_NEXT_OF_INNER_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[C_RELEASE_NEXT_OF_INNER_RAW:%.*]] = arith.addi [[C_INNER_LOOP]]#0, [[C_REL_ONE]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[C_REL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 3 : i32
      // CHECK: [[C_RELEASE_NEXT_OF_INNER_REM:%.*]] = arith.remsi [[C_RELEASE_NEXT_OF_INNER_RAW]], [[C_REL_DEPTH]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[C_RELEASE_NEXT_OF_INNER_STAGE:%.*]] = arith.select {{.*}}, {{.*}}, [[C_RELEASE_NEXT_OF_INNER_REM]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: nvws.semaphore.release [[C_ENTRY]][[[C_RELEASE_NEXT_OF_INNER_STAGE]]], [[C_EXIT_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %inner : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }

  // D: the acquire preceding the nonempty inner loop is the fresh epoch.
  // CHECK-LABEL: @case_d
  tt.func @case_d(%lb: i32, %ub: i32, %step: i32,
                  %lhs: !ttg.memdesc<128x64xf32, #shared32, #smem>,
                  %rhs: !ttg.memdesc<64x128xf32, #shared32, #smem>) {
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    // CHECK: [[D_BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 3 : i32}
    // CHECK: [[D_ENTRY:%.*]] = nvws.semaphore.create [[D_BASE]] released = 1 {pending_count = 1 : i32}
    // CHECK: [[D_NEXT:%.*]] = nvws.semaphore.create [[D_BASE]] released = 6 {pending_count = 1 : i32}
    // CHECK: [[D_INITIAL_CURRENT_STAGE:%.*]] = arith.constant 2 : i32
    // CHECK: [[D_INITIAL_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 1 : i32
    // CHECK: [[D_INITIAL_NEXT_RAW:%.*]] = arith.addi [[D_INITIAL_CURRENT_STAGE]], [[D_INITIAL_ONE]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // CHECK: [[D_INITIAL_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 3 : i32
    // CHECK: [[D_INITIAL_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_INITIAL_NEXT_RAW]], [[D_INITIAL_DEPTH]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // CHECK: [[D_INITIAL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} 0 : i32
    // CHECK: [[D_INITIAL_NEXT_STAGE:%.*]] = arith.select [[D_INITIAL_NEEDS_WRAP]], [[D_INITIAL_ZERO]], [[D_INITIAL_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 13 : i32} : i32
    // CHECK: [[D_INIT_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_INITIAL_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 13 : i32}
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for {{.*}} iter_args([[D_W0_TOK:%.*]] = [[D_INIT_TOK]], [[D_CURRENT_STAGE:%.*]] = [[D_INITIAL_NEXT_STAGE]],
    %outer = scf.for %i = %lb to %ub step %step iter_args(%outer_token = %tok) -> (!ttg.async.token) : i32 {
      // CHECK: [[D_W0_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{%.*}}, [[D_W0_BUF]][], {{%.*}} {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[D_NEXT]][[[D_CURRENT_STAGE]]], [[D_W0_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %w0 = ttng.tmem_store %zero, %acc[%outer_token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[D_NEXT_ONE:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
      // CHECK: [[D_NEXT_RAW:%.*]] = arith.addi [[D_CURRENT_STAGE]], [[D_NEXT_ONE]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[D_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 3 : i32
      // CHECK: [[D_NEEDS_WRAP:%.*]] = arith.cmpi eq, [[D_NEXT_RAW]], [[D_DEPTH]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[D_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
      // CHECK: [[D_NEXT_STAGE:%.*]] = arith.select [[D_NEEDS_WRAP]], [[D_ZERO]], [[D_NEXT_RAW]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[D_W1_TOK:%.*]] = nvws.semaphore.acquire [[D_NEXT]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: scf.for
      %inner = scf.for %j = %lb to %ub step %step iter_args(%inner_token = %w0) -> (!ttg.async.token) : i32 {
        // CHECK: [[D_W1_BUF:%.*]] = nvws.semaphore.buffer [[D_NEXT]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] {ttg.partition = array<i32: 1>}
        // CHECK: ttng.tc_gen5_mma {{%.*}}, {{%.*}}, [[D_W1_BUF]][], {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>}
        %w1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%inner_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared32, #smem>, !ttg.memdesc<64x128xf32, #shared32, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1>} %w1 : !ttg.async.token
      } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_W1_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[D_READ_TOK:%.*]] = nvws.semaphore.acquire [[D_ENTRY]][[[D_NEXT_STAGE]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[D_READ_BUF:%.*]] = nvws.semaphore.buffer [[D_ENTRY]][[[D_NEXT_STAGE]]], [[D_READ_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK: {{%.*}}, {{%.*}} = ttng.tmem_load [[D_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %value, %read = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_d"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %read : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}

// -----

// Meta flash attention forward (persistent) IR, captured from
// meta-aws-logs/run-22may26-nvws-meta-tmem-crash/passes/
// 062-anonymous_VerifyWarpSpecializationPartitions.mlir (loc() stripped).
// High-coverage exercise of native point-of-use insert-semas with a persistent
// outer loop wrapping a pipelined inner loop. Native POU opens buffer.id=5 at
// its first inner-loop use and closes the recurrence with releases from the
// stage-1 uses; no buffer.id=5 token is carried through either loop.
// buffer.id=4 stays behind its own in-loop gate. The
// buffer.id=2/3 per-iteration accumulators thread tokens through the outer
// loop only (bottom acquire at the final readout) and re-acquire in-body
// inside the inner loop. The Q/K/V descriptor SMEM, stats stores, and epilogue
// O buffers exercise the remaining protocols. Hand-curated, semaphore-focused
// CHECKs.
//
// NOTE: the pass canonicalizes the TMEM encoding aliases, so in the captured
// output #tmem is blockN=128 (the 128x128 acc/result/f16-view buffers) and
// #tmem1 is blockN=1 (the alpha/l_i stats subslices) — the opposite of the
// input aliases below. CHECK lines therefore use the *output* alias names.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL:     tt.func public @_attn_fwd_persist(
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %Z, %n_tile_num : i32
    %total_tiles_3 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs : i32
    %0 = arith.remsi %total_tiles_3, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_19 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_19 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %desc_q_4 = arith.muli %Z, %H : i32
    %desc_q_5 = arith.muli %desc_q_4, %c16384_i32 : i32
    %desc_q_6 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_q_7 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_k_8 = tt.make_tensor_descriptor %desc_k, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_v_9 = tt.make_tensor_descriptor %desc_v, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_o_10 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %desc_o_11 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<128x128xf16, #shared>
    %offset_y = arith.muli %H, %c16384_i32 : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m0_12 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_ij_13 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %qk_14 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>

    // The single-buffer Q/K/V SMEM allocs each get a true(EMPTY)/false(FULL)
    // semaphore pair (pending_count = 1).
    %q0_0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %q0_1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The buffer.id=4 accumulator-class TMEM allocation is hoisted to a single
    // 1x128x128 alloc whose subslices (alpha/offsetkv stats + acc + f16 view)
    // form one multi-member semaphore group: two released gates (pending_count =
    // 2) plus five false(FULL) phases (pending_count = 1).
    %alpha = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %alpha_15 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_16 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // The buffer.id=5 accumulator-class TMEM allocation forms the second
    // multi-member group: two released gates (pending_count = 2) plus five
    // false(FULL) semaphores (pending_count = 1).
    %offsetkv_y_17 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_18 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // Epilogue-store SMEM scratch (%3,%4) each get a true/false pair.
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // The two per-iteration acc TMEM allocs (buffer.id 2/3) are hoisted to
    // single-component 1x buffers with a true/false pair each.

    // Initial outer-gate acquires for R4/R5 (partition 1), followed by the two
    // per-iteration accumulator tokens. Only ACC0/ACC1 thread through the
    // outer loop; the R5 token does not.

    // R5 is not carried through the outer persistent loop. R4 also needs no
    // carried token.
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_19 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_hz = arith.divsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_z = arith.divsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_h = arith.remsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_20 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_21 = arith.muli %off_h, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %5 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %q0 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_m0_24 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_25 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_26 = arith.addi %offs_m0_24, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m0_27 = arith.addi %offs_m0_25, %offs_m0_12 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>

      // Q0 load: acquire EMPTY, point-of-use buffer feeds the descriptor_load,
      // release FULL with a tma_load arrive.
      nvws.descriptor_load %desc_q_6[%qo_offset_y_23, %c0_i32] 32768 %q0_0 {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Q1 load mirrors Q0.
      nvws.descriptor_load %desc_q_7[%q0, %c0_i32] 32768 %q0_1 {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qk_0, %qk_0_32 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_1, %qk_1_33 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 4>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // The acc_0 / acc_1 init stores reuse the carried ACC0/ACC1 EMPTY tokens
      // (no re-acquire here), write zero into the point-of-use buffer, then
      // release the EMPTY gate for the first in-body acquire in the inner loop.
      %acc_0, %acc_0_34 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_1, %acc_1_35 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // Outside the inner loop, the K/V FULL tokens consumed by the trailing
      // tc5mma releases are acquired up front (partition 1) on the Q FULL sems.

      // R5 is not carried through the inner pipelined loop either. The R4
      // inner gate and Q FULL tokens are loop-invariant captures; ACC0/ACC1
      // are re-acquired in-body.
      %offsetkv_y_40:9 = scf.for %offsetkv_y_88 = %c0_i32 to %c16384_i32 step %c128_i32 iter_args(%offset_y_89 = %offset_y_22, %arg12 = %cst_2, %arg13 = %cst_1, %qk_0_90 = %qk_0_32, %acc_91 = %acc_0_34, %arg16 = %cst_2, %arg17 = %cst_1, %qk_1_92 = %qk_1_33, %acc_93 = %acc_1_35) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        // K descriptor load: acquire EMPTY (K_E), point-of-use buffer, release FULL.
        nvws.descriptor_load %desc_k_8[%offset_y_89, %c0_i32] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_97 = ttg.memdesc_reinterpret %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_98 = ttg.memdesc_trans %k_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        // V descriptor load: acquire EMPTY (V_E), point-of-use buffer, release FULL.
        nvws.descriptor_load %desc_v_9[%offset_y_89, %c0_i32] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // First QK MMA (partition 1): acquire and buffer the R4 inner gate,
        // buffer the Q0 FULL token (lhs), then acquire+buffer the inner K FULL
        // token (rhs, transposed). MMA lhs is the Q0 buffer, acc is R4_QK#3.
        %qk_101 = ttng.tc_gen5_mma %q0_0, %k_98, %qk_0[%qk_0_90], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // Second QK MMA mirrors the first on the R5 (id=5) acc-class set. Its
        // R5 token is acquired at this first use, not carried into the loop.
        // After it, the inner K_E EMPTY is released using the K consumer token.
        %qk_102 = ttng.tc_gen5_mma %q0_1, %k_98, %qk_1[%qk_1_92], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // QK tmem_load (partition 5): acquire R4_F1, buffer, load #3.
        %qk_103, %qk_104 = ttng.tmem_load %qk_0[%qk_101] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_105 = ttg.convert_layout %qk_103 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        // QK tmem_load (partition 4): acquire R5_F1, buffer, load #3.
        %qk_106, %qk_107 = ttng.tmem_load %qk_1[%qk_102] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_108 = ttg.convert_layout %qk_106 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %m_ij_109 = "tt.reduce"(%qk_105) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_110 = "tt.reduce"(%qk_108) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_111 = arith.mulf %m_ij_109, %m_ij {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_112 = arith.mulf %m_ij_110, %m_ij_13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_113 = arith.maxnumf %arg13, %m_ij_111 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_114 = arith.maxnumf %arg17, %m_ij_112 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_115 = arith.mulf %qk_105, %qk {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_116 = arith.mulf %qk_108, %qk_14 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %qk_117 = tt.expand_dims %m_ij_113 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_118 = tt.expand_dims %m_ij_114 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_119 = tt.broadcast %qk_117 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_120 = tt.broadcast %qk_118 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_121 = arith.subf %qk_115, %qk_119 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_122 = arith.subf %qk_116, %qk_120 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_121 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %p_123 = math.exp2 %qk_122 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %alpha_124 = arith.subf %arg13, %m_ij_113 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_125 = arith.subf %arg17, %m_ij_114 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_126 = math.exp2 %alpha_124 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_127 = tt.expand_dims %alpha_126 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_128 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} true
        // alpha stats store (partition 5) writes into the R4 subslice #0 buffer
        // already opened by [[QK0_BUF]] (point-of-use), then releases R4 FULL.
        ttng.tmem_store %alpha_127, %alpha, %alpha_128 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %alpha_129 = math.exp2 %alpha_125 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_130 = tt.expand_dims %alpha_129 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_131 = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} true
        // alpha_15 stats store (partition 4) mirrors on R5 subslice #0.
        ttng.tmem_store %alpha_130, %alpha_15, %alpha_131 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij_132 = "tt.reduce"(%p_123) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        // acc_0 tmem_load (partition 0) re-acquires ACC0 EMPTY in-body and
        // buffers under the fresh token.
        %acc_133, %acc_134 = ttng.tmem_load %acc_0[%acc_91] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %acc_135, %acc_136 = ttng.tmem_load %acc_1[%acc_93] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %18 = tt.reshape %acc_133 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %19 = tt.reshape %acc_135 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %20 = tt.trans %18 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %21 = tt.trans %19 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %20 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_137, %outRHS_138 = tt.split %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        // alpha stats reload (partition 0): acquire R4_F2, buffer, load subslice
        // #0, release the second arrival to R4_IN.
        %alpha_139, %alpha_140 = ttng.tmem_load %alpha[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_141 = tt.reshape %alpha_139 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_142 = ttg.convert_layout %alpha_141 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_143 = tt.expand_dims %alpha_142 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        // alpha_15 stats reload (partition 0): acquire R5_F2, buffer, load, and
        // release the second arrival to R5_IN for the next iteration's POU
        // acquire.
        %alpha_144, %alpha_145 = ttng.tmem_load %alpha_15[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_146 = tt.reshape %alpha_144 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_147 = ttg.convert_layout %alpha_146 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_148 = tt.expand_dims %alpha_147 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_149 = ttg.convert_layout %acc0_143 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_150 = ttg.convert_layout %acc0_148 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_151 = tt.broadcast %acc0_149 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_152 = tt.broadcast %acc0_150 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_153 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_154 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_137, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1_155 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_138, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc_156 = tt.join %acc0_153, %acc1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_157 = tt.join %acc0_154, %acc1_155 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_158 = tt.trans %acc_156 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_159 = tt.trans %acc_157 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_160 = tt.reshape %acc_158 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %acc_161 = tt.reshape %acc_159 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %p_162 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %p_163 = arith.truncf %p_123 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        // p (f16) store into the R4 f16 view subslice #4 (partition 5) reuses
        // the retained R4_F1 token and releases R4_F3.
        %acc_164 = ttng.tmem_alloc %p_162 {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // p_123 (f16) store into R5 f16 view subslice #4 (partition 4) reuses
        // the retained R5_F1 token and releases R5_F3.
        %acc_165 = ttng.tmem_alloc %p_163 {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        // acc_0 update store (partition 0) into the in-body acquired ACC0
        // buffer, release ACC0_F on the same token.
        %acc_166 = ttng.tmem_store %acc_160, %acc_0[%acc_134], %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_167 = ttng.tmem_store %acc_161, %acc_1[%acc_136], %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #1 (partition 1): buffer R4_F3 (p f16 view #4), acquire ACC0_F
        // FULL, acquire V_F FULL, tc5mma, release ACC0_E.
        %acc_170 = ttng.tc_gen5_mma %acc_164, %v, %acc_0[%acc_166], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        // PV MMA #2 (partition 1): R5_F3 / ACC1_F / re-use V buffer, release the
        // inner V_E EMPTY and ACC1_E.
        %acc_171 = ttng.tc_gen5_mma %acc_165, %v, %acc_1[%acc_167], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg12, %alpha_126 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_172 = arith.mulf %arg16, %alpha_129 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_173 = arith.addf %l_i0, %l_ij {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_174 = arith.addf %l_i0_172, %l_ij_132 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_175 = arith.addi %offset_y_89, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
        // No R5 token is re-acquired at the loop boundary or yielded.
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %offsetkv_y_175, %l_i0_173, %m_ij_113, %qk_104, %acc_170, %l_i0_174, %m_ij_114, %qk_107, %acc_171 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token
      // Inner-loop close: pinned pipelining attrs.
      } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>]}

      // Post-inner-loop epilogue (still inside the persistent outer loop). The
      // inner Q FULL tokens release back as EMPTY for the next outer iteration.
      // R5 is acquired at the post-inner use, while R4 uses its point-of-use
      // drain; each opens one final FULL semaphore.

      // Both post-loop stats stores (partition 4) land in the single R5_F4
      // phase: one acquire, one buffer, store #2 then #1.
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y_40#6 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_42 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_41, %offsetkv_y_18, %offsetkv_y_42 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y_17 stats store (partition 4) reuses the same token for
      // buffer #1, then releases R5_F5 and the first R5_E arrival.
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y_40#5 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_44 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_43, %offsetkv_y_17, %offsetkv_y_44 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>

      // Both post-loop stats stores (partition 5) land in the single R4_F4
      // phase: acquire, buffer, store #2 then #1.
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y_40#2 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_46 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_45, %offsetkv_y_16, %offsetkv_y_46 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y stats store (partition 5) reuses the same token for buffer
      // #1, then releases R4_F5 and the first R4_E arrive.
      %offsetkv_y_47 = tt.expand_dims %offsetkv_y_40#1 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_48 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_47, %offsetkv_y, %offsetkv_y_48 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      // offsetkv_y reload (partition 0): acquire R4_F5, buffer, load #1.
      %offsetkv_y_49, %offsetkv_y_50 = ttng.tmem_load %offsetkv_y[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_51 = tt.reshape %offsetkv_y_49 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_52 = ttg.convert_layout %offsetkv_y_51 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0 = math.log2 %offsetkv_y_52 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_17 reload (partition 0): acquire R5_F5, buffer, load #1.
      %offsetkv_y_53, %offsetkv_y_54 = ttng.tmem_load %offsetkv_y_17[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_55 = tt.reshape %offsetkv_y_53 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_56 = ttg.convert_layout %offsetkv_y_55 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_57 = math.log2 %offsetkv_y_56 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_16 reload (partition 0): load #2 from the R4_F5 buffer,
      // then the second R4_E arrive.
      %offsetkv_y_58, %offsetkv_y_59 = ttng.tmem_load %offsetkv_y_16[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_60 = tt.reshape %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_61 = ttg.convert_layout %offsetkv_y_60 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_62 = arith.addf %offsetkv_y_61, %m_i0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      // offsetkv_y_18 reload (partition 0): load #2 from the R5_F5 buffer,
      // then the second R5_E arrive.
      %offsetkv_y_63, %offsetkv_y_64 = ttng.tmem_load %offsetkv_y_18[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_65 = tt.reshape %offsetkv_y_63 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_66 = ttg.convert_layout %offsetkv_y_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_67 = arith.addf %offsetkv_y_66, %m_i0_57 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = tt.expand_dims %offsetkv_y_52 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_68 = tt.expand_dims %offsetkv_y_56 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_69 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc0_70 = tt.broadcast %acc0_68 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      // Final acc_0 readout (partition 0): bottom re-acquire of ACC0 EMPTY
      // (carried to the next outer iteration via yield), buffer, load.
      %acc, %acc_71 = ttng.tmem_load %acc_0[%offsetkv_y_40#4] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_72 = ttg.convert_layout %acc {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      // Final acc_1 readout (partition 0) mirrors acc_0.
      %acc_73, %acc_74 = ttng.tmem_load %acc_1[%offsetkv_y_40#8] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_75 = ttg.convert_layout %acc_73 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_76 = arith.divf %acc_72, %acc0_69 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %acc0_77 = arith.divf %acc_75, %acc0_70 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %m_ptrs0 = arith.muli %off_hz, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %m_ptrs0_78 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : !tt.ptr<f32>, i32
      %m_ptrs0_79 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_80 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_81 = tt.addptr %m_ptrs0_79, %offs_m0_26 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %m_ptrs0_82 = tt.addptr %m_ptrs0_80, %offs_m0_27 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %6 = ttg.convert_layout %m_i0_62 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %7 = ttg.convert_layout %m_i0_67 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      tt.store %m_ptrs0_81, %6 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %m_ptrs0_82, %7 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      // Epilogue O0 local_store (partition 0): acquire O0 EMPTY, point-of-use
      // buffer, store, release O0 FULL.
      %8 = arith.truncf %acc0_76 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %8, %3 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O1 local_store (partition 0) mirrors O0.
      %10 = arith.truncf %acc0_77 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      ttg.local_store %10, %4 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // Epilogue O0 local_load (partition 2): acquire O0 FULL, buffer and
      // load. Ownership remains live through its descriptor store below.
      %13 = ttg.local_load %3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %14 = ttg.convert_layout %13 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      // Epilogue O1 local_load (partition 2) mirrors O0. Each empty release
      // follows the descriptor store that completes that channel's read.
      %16 = ttg.local_load %4 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      %17 = ttg.convert_layout %16 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_10[%qo_offset_y_23, %c0_i32], %14 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_11[%5, %c0_i32], %17 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, tensor<128x128xf16, #blocked1>
      %tile_idx_87 = arith.addi %tile_idx_19, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      // Outer-loop end bridges R4 and R5 back to their point-of-use gates
      // (two arrivals each). Only the ACC0/ACC1 readout tokens ride the yield.
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tile_idx_87 : i32
    // Outer-loop close: pinned warp-specialize attrs (stages, tag, types).
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4, 5>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// These tests cover slot replay across an atomic scheduled region.  The
// scf.if itself owns the stage-2 schedule; its child operations intentionally
// have no loop.stage/loop.cluster attributes.

#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked128 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance the depth-5 cursor from A's slot to B's slot.
  // The conditional C wave is a same-owner overwrite after B has been read,
  // so it reuses B's slot and does not advance the cursor.  The region-closing
  // release therefore supplies A three logical iterations later:
  //
  //   A(i) = 2i mod 5, B/C(i) = 2i+1 mod 5, A(i+3) = B/C(i).
  //
  // The only authored non-zero displacement is the A-read to B-write handoff.
  // The scf.if returns an owner-0 token: the then-branch hands the C-read
  // token back to owner 0, while the else-branch passes the B-read token
  // through.  One common ENTRY release follows the conditional.
  // CHECK-LABEL: @depth5_regular_atomic_if
  tt.func @depth5_regular_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                    %cond: i1) {
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = 21 {pending_count = 1 : i32}
    // CHECK: [[A_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[A_TO_B:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[B_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // No token iter_args: every acquire is at its point of use in the body.
    // After ASP the loop carries the cursor plus phase words, all i32.
    // CHECK: %{{[0-9]+}}:7 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[A_SLOT:%.*]] = arith.select {{.*}} : i32
      // CHECK: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[A_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[A_SLOT]]], [[A_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[A_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[A_FULL]][[[A_SLOT]]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // The A read hands its slot off to the B write at displacement +1.
      // CHECK: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[A_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[A_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_FULL]][[[A_SLOT]]], [[A_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[A_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[TO_B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TO_B_REM:%.*]] = arith.remsi [[TO_B_RAW]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TO_B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[TO_B_REM]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      // CHECK: nvws.semaphore.release [[A_TO_B]][[[TO_B_SLOT]]], [[A_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_a"(%a_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // B's fresh acquire advances the cursor to [[B_SLOT]] = [[A_SLOT]] + 1.
      // CHECK: [[B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[B_RAW]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[B_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[A_TO_B]][[[B_SLOT]]], [[B_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[B_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[B_FULL]][[[B_SLOT]]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[B_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[B_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_b"(%b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // CHECK: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C overwrites B's slot under the still-held B-read token: it is
        // rendered through the C member of B_FULL's buffer tuple, so the
        // atomic region does not add a third cursor advance.
        // CHECK: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]] {ttg.partition = array<i32: 0>}
        // CHECK: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[C_FULL]][[[B_SLOT]]], [[B_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // CHECK: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // CHECK: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[B_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // CHECK: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[IF_BACK]][[[B_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[B_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // CHECK: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[B_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the B-read token through.
        // CHECK: scf.yield {{.*}}[[B_READ_TOK]], [[B_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // CHECK: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1>, array<i32: 0>, array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Consecutive A/B writes by the same owner share one token and therefore
  // make one fresh depth-3 cursor advance.  Both reads and the atomic C region
  // use that stage.  The scf.if returns an owner-0 token from either the C read
  // or the unchanged A/B-read path.  One common release then returns the slot
  // to A after one full three-iteration orbit.
  // CHECK-LABEL: @depth3_same_owner_atomic_if
  tt.func @depth3_same_owner_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                       %cond: i1) {
    // CHECK: [[ENTRY:%.*]] = nvws.semaphore.create {{.*}} released = 7 {pending_count = 1 : i32}
    // CHECK: [[AB_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[C_FULL:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    // CHECK: [[IF_BACK:%.*]] = nvws.semaphore.create {{.*}} {pending_count = 1 : i32}
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    // CHECK: %{{[0-9]+}}:5 = scf.for {{.*}} iter_args([[CURSOR:%.*]] = %{{[-A-Za-z0-9_.$#]+}},
    scf.for %iv = %lb to %ub step %step : i32 {
      // Both writes share the single ENTRY acquire of this iteration.
      // CHECK: [[AB_SLOT:%.*]] = arith.select {{.*}} : i32
      // CHECK: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[ENTRY]][[[AB_SLOT]]], [[AB_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_store {{.*}}, [[AB_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[AB_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[AB_FULL]][[[AB_SLOT]]], [[AB_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Both reads share the single AB_FULL acquire.
      // CHECK: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[AB_SLOT]], {{%.*}}] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[AB_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_load [[AB_READ_VIEWS]]#0 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      // CHECK: ttg.local_load [[AB_READ_VIEWS]]#1 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_ab"(%a_read, %b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>, tensor<128x64xf16, #blocked64>) -> ()
      // The scf.if yields its owner-0 token.  ASP also returns the shared slot
      // and the phase words updated inside the branch.
      // CHECK: [[IF_RESULTS:%.*]]:4 = scf.if
      scf.if %cond {
        // C is rendered through the C member of AB_FULL's buffer tuple: the
        // atomic region reuses the shared slot without a cursor advance.
        // CHECK: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] {ttg.partition = array<i32: 0>}
        // CHECK: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[C_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        // The C read hands ownership back to owner 0 before the branch yields.
        // CHECK: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 1>}
        // CHECK: [[C_READ_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[C_FULL]][[[AB_SLOT]]], [[C_READ_TOK]] {ttg.partition = array<i32: 1>}
        // CHECK: ttg.local_load [[C_READ_VIEWS]]#2 {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[IF_BACK]][[[AB_SLOT]]], [[C_READ_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[IF_THEN_TOKEN:%.*]] = nvws.semaphore.acquire [[IF_BACK]][[[AB_SLOT]], {{%.*}}] {ttg.partition = array<i32: 0>}
        // CHECK: scf.yield {{.*}}[[IF_THEN_TOKEN]], [[AB_SLOT]],
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
        // Without the C wave, the branch passes the A/B-read token through.
        // CHECK: scf.yield {{.*}}[[AB_READ_TOK]], [[AB_SLOT]],
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // CHECK: } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: scf.yield {{.*}}[[IF_RESULTS]]#1,
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}

// -----
