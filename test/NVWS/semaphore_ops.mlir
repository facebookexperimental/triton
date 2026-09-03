// RUN: triton-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_roundtrip
  tt.func @semaphore_roundtrip(%buf : !ttg.memdesc<3x64x16xf16, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create {{%.*}} released = 5 {pending_count = 2 : i32}
    %sem = nvws.semaphore.create %buf released = 5 {pending_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared, #smem>]>
    // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][{{%.*}}, {{%.*}}]
    %tok = nvws.semaphore.acquire %sem[%c1, %c0] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared, #smem>]> -> !ttg.async.token
    // CHECK: nvws.semaphore.buffer [[SEM]][{{%.*}}], [[TOK]]
    %view = nvws.semaphore.buffer %sem[%c1], %tok : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[SEM]][{{%.*}}], [[TOK]] [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] {arrive_count = 1 : i32}
    nvws.semaphore.release %sem[%c1], %tok [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @duplicate_async_kind() {
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0, %c0] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{async_ops contains duplicate async kind}}
    nvws.semaphore.release %sem[%c0], %tok [#nvws.async_op<none>, #nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @released_mask_outside_depth() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // expected-error @below {{released_mask has bits outside semaphore depth 3}}
    %sem = nvws.semaphore.create %buf released = 8 : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    tt.return
  }
}
