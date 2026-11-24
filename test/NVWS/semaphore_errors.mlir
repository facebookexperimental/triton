// RUN: split-file %s %t
// RUN: triton-opt %t/ir.mlir --split-input-file --allow-unregistered-dialect --verify-diagnostics
// RUN: not triton-opt %t/assign-partition-overlap.mlir --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase 2>&1 | FileCheck %t/assign-partition-overlap.mlir --check-prefix=PARTITION-OVERLAP
// RUN: not triton-opt %t/assign-split-partition-overlap.mlir --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase 2>&1 | FileCheck %t/assign-split-partition-overlap.mlir --check-prefix=SPLIT-PARTITION-OVERLAP
// RUN: not triton-opt %t/assign-depth-limit.mlir --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase 2>&1 | FileCheck %t/assign-depth-limit.mlir --check-prefix=DEPTH-LIMIT
// RUN: triton-opt %t/lower-count-errors.mlir --split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore --verify-diagnostics
// RUN: triton-opt %t/lower-requires-assignment.mlir --allow-unregistered-dialect --nvws-lower-semaphore --verify-diagnostics

//--- ir.mlir

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_release_duplicate_async() {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{async_ops contains duplicate async kind}}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>, #nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_leading_dims_mismatch(%d : !ttg.memdesc<1x1xi32, #shared0, #smem>, %e : !ttg.memdesc<2x1xi32, #shared0, #smem>) {
    // expected-error @below {{Leading dims of sliced semaphore inputs don't match}}
    %sem = nvws.semaphore.create %d, %e released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem>, !ttg.memdesc<2x1xi32, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_buffer_used_elsewhere(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>) {
    // expected-error @below {{Semaphore buffer is used elsewhere, Semaphore cannot guarantee async safety}}
    %sem = nvws.semaphore.create %d released = 1 : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>]>
    %tmp = ttng.tmem_alloc %d : (!ttg.memdesc<1x64x16xf16, #shared0, #smem>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_partial_overlap_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %a, %c : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %c : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_permuted_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %b, %a : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_arity_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Semaphore has different number of arguments than buffer}}
    %views:2 = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_dimensions_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Dimensions don't match}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_result_must_be_mutable() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Semaphore buffer result memdesc must be mutable}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared0, #smem>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_released_mask_outside_depth() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared0, #smem, mutable>
    // expected-error @below {{released_mask has bits outside semaphore depth 3}}
    %sem = nvws.semaphore.create %buf released = 8 : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared0, #smem, mutable>]>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @semaphore_backing_view_chain_rejects_non_protocol_use() {
    %base = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %base {offset = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %view = ttg.memdesc_reinterpret %sub : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    "use"(%view) : (!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>) -> ()
    // expected-error @+1 {{Semaphore buffer is used elsewhere, Semaphore cannot guarantee async safety}}
    %empty = nvws.semaphore.create %base, %view released = 1 : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    "use_sema"(%empty) : (!nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>]>) -> ()
    tt.return
  }
}

//--- assign-partition-overlap.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // PARTITION-OVERLAP: error: circular semaphore has overlapping cross-partition physical-slot ownership
  tt.func @reject_overlapping_partition_owned_slots(%lb: i32, %ub: i32,
                                                     %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = 306 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base released = 7 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %a0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %ta = nvws.semaphore.acquire %empty[%a0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %ba = nvws.semaphore.buffer %empty[%a0], %ta {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%ba) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %b0 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tb = nvws.semaphore.acquire %empty[%b0] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bb = nvws.semaphore.buffer %empty[%b0], %tb {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bb) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    } {tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }
}

//--- assign-split-partition-overlap.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // D = 4, A = 2, and G = 2.  X owns class 1 in partition 1.
  // Y and Z both own class 0, but from partitions 2 and 1 respectively.
  // The per-partition phase-split proof accepts X/Z as disjoint; the global
  // partition-ownership proof must reject the Y/Z overlap.
  // SPLIT-PARTITION-OVERLAP: error: circular semaphore has overlapping cross-partition physical-slot ownership
  tt.func @reject_split_phase_cross_partition_slot_overlap(
      %lb: i32, %ub: i32, %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 307 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<4x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %base released = 15 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>
    %driver = nvws.semaphore.create %base released = 15 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %zd0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td0 = nvws.semaphore.acquire %driver[%zd0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd0 = nvws.semaphore.buffer %driver[%zd0], %td0 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd0) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %x = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tx = nvws.semaphore.acquire %sem[%x] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %zd1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %td1 = nvws.semaphore.acquire %driver[%zd1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %bd1 = nvws.semaphore.buffer %driver[%zd1], %td1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%bd1) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()

      %y = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %ty = nvws.semaphore.acquire %sem[%y] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token

      %z = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %tz = nvws.semaphore.acquire %sem[%z] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<4x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}

//--- assign-depth-limit.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // DEPTH-LIMIT: error: multiphase bitmask supports at most 32 buffer stages, got 33
  tt.func @reject_phase_mask_depth_above_32() {
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<33x1xi32, #shared, #smem, mutable>
    %semaphore = nvws.semaphore.create %buffer released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<33x1xi32, #shared, #smem, mutable>]>
    tt.return
  }
}

//--- lower-count-errors.mlir

// Negative coverage for the first-class count contract: lowering requires
// authored counts and arrive multiplicity is only lowerable for sync kinds.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @create_missing_pending_count() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // expected-error @below {{semaphore.create reached nvws-lower-semaphore without a pending_count; the producing pass must author it}}
    %sem = nvws.semaphore.create %buf released = 3 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @release_missing_arrive_count() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{semaphore.release reached nvws-lower-semaphore without an arrive_count; the producing pass must author it}}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @arrive_count_two_with_async_kind() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 3 {pending_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{arrive_count > 1 is only lowerable for none/wgmma async kinds}}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 0>, arrive_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

//--- lower-requires-assignment.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @requires_assignment() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // expected-error @below {{requires stage and phase operands; run nvws-assign-semaphore-stage-phase first}}
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{requires a stage operand; run nvws-assign-semaphore-stage-phase first}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // expected-error @below {{requires a stage operand; run nvws-assign-semaphore-stage-phase first}}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    "use"(%view) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
