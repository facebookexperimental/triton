// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_create
  // CHECK: nvws.semaphore.create {{.*}} released = 1
  // CHECK: nvws.semaphore.create {{.*}}
  tt.func @semaphore_create(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.semaphore.create %d, %e released = 1 : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.create %d : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_roundtrip
  tt.func @semaphore_roundtrip(%buf : !ttg.memdesc<3x64x16xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create {{%.*}} released = 5 {pending_count = 2 : i32}
    %sem = nvws.semaphore.create %buf released = 5 {pending_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][{{%.*}}, {{%.*}}]
    %tok = nvws.semaphore.acquire %sem[%c1_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    // CHECK: nvws.semaphore.buffer [[SEM]][{{%.*}}], [[TOK]]
    %view = nvws.semaphore.buffer %sem[%c1_i32], %tok : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>
    // CHECK: nvws.semaphore.release [[SEM]][{{%.*}}], [[TOK]] [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] {arrive_count = 1 : i32}
    nvws.semaphore.release %sem[%c1_i32], %tok [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_acquire_buffer
  // CHECK: nvws.semaphore.create {{.*}} released = 5
  // CHECK: nvws.semaphore.acquire {{.*}} : <[{{.*}}]> -> !ttg.async.token
  // CHECK: nvws.semaphore.acquire {{.*}}[{{.*}}, {{.*}}] : <[{{.*}}]> -> !ttg.async.token
  // CHECK: nvws.semaphore.buffer {{.*}}, {{.*}} : <[{{.*}}]>, !ttg.async.token ->
  // CHECK: nvws.semaphore.buffer {{.*}}[{{.*}}], {{.*}} : <[{{.*}}]>, !ttg.async.token ->
  tt.func @semaphore_acquire_buffer(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = nvws.semaphore.create %d released = 7 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %mask = nvws.semaphore.create %d released = 5 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.acquire %0 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    %2 = nvws.semaphore.acquire %0[%c1_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %1 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>
    %4 = nvws.semaphore.buffer %0[%c1_i32], %2 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_release
  // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
  // CHECK: nvws.semaphore.release {{.*}}[{{.*}}], {{.*}} [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] : <[{{.*}}]>, !ttg.async.token
  tt.func @semaphore_release(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = nvws.semaphore.create %d : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.acquire %0 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    nvws.semaphore.release %0, %1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token
    nvws.semaphore.release %0[%c0_i32], %1 [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}
