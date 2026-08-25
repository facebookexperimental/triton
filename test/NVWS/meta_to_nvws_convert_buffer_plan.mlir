// RUN: triton-opt %s --nvws-meta-to-nvws-convert | FileCheck %s \
// RUN:   --implicit-check-not=ttg.memdesc_reinterpret \
// RUN:   --implicit-check-not=allocation.reuseTarget \
// RUN:   --implicit-check-not="buffer.id = 22"
// RUN: triton-opt %s --nvws-meta-to-nvws-convert --nvws-meta-to-nvws-convert | FileCheck %s \
// RUN:   --implicit-check-not=ttg.memdesc_reinterpret \
// RUN:   --implicit-check-not=allocation.reuseTarget \
// RUN:   --implicit-check-not="buffer.id = 22"

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @buffer_plan
  tt.func @buffer_plan(%value: tensor<64x64xf16, #blocked>, %lb: i32,
                       %ub: i32, %step: i32) {
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[HOST:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 3 : i32, buffer.start = 0 : i32}
    %reuse0 = ttg.local_alloc {allocation.reuseTarget = 3 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 3 : i32, buffer.start = 1 : i32}
    %reuse1 = ttg.local_alloc {allocation.reuseTarget = 3 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32}
    %single = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 23 : i32}
    // CHECK-NOT: buffer.circular
    // CHECK-NOT: buffer.start
    %incompatible = ttg.local_alloc {allocation.reuseTarget = 9 : i32, buffer.copy = 1 : i32, buffer.id = 23 : i32} : () -> !ttg.memdesc<64x64xf16, #shared64, #smem, mutable>
    // CHECK-NOT: async_task_id
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[PLANNED:.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 24 : i32, ttg.partition = array<i32: 2>}
      %planned = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 24 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, %[[PLANNED]] {ttg.partition = array<i32: 2>}
      ttg.local_store %value, %planned {async_task_id = array<i32: 2>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load %[[PLANNED]] {ttg.partition = array<i32: 2>}
      %loaded = ttg.local_load %planned {async_task_id = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "gemm"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @tmem_plan
  tt.func @tmem_plan(%value: tensor<128x128xf32, #acc>, %lb: i32,
                     %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %true = arith.constant {async_task_id = array<i32: 2>} true
      // CHECK: %[[TMEM:.*]], %[[TOKEN:.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 25 : i32, ttg.partition = array<i32: 2>}
      %buffer, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 25 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: %[[STORED:.*]] = ttng.tmem_store %{{.*}}, %[[TMEM]][%[[TOKEN]]], %{{.*}} {ttg.partition = array<i32: 2>}
      %stored = ttng.tmem_store %value, %buffer[%token], %true {async_task_id = array<i32: 2>} : tensor<128x128xf32, #acc> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load %[[TMEM]][%[[STORED]]] {ttg.partition = array<i32: 2>}
      %loaded, %load_token = ttng.tmem_load %buffer[%stored] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc>
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "gemm"],
       ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @localize_managed_groups
  tt.func @localize_managed_groups(
      %early: i1, %value: tensor<1xi32, #blocked1>,
      %accValue: tensor<128x128xf32, #acc>, %lb: i32, %ub: i32,
      %step: i32) {
    // The complete source-free groups move together; none remain before the
    // early-return branch after conversion.
    // CHECK-NOT: buffer.id = 30
    // CHECK-NOT: buffer.id = 31
    // CHECK: cf.cond_br
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 30 : i32} : () -> !ttg.memdesc<2xi32, #shared1, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 30 : i32} : () -> !ttg.memdesc<2xi32, #shared1, #smem, mutable>
    %tmem, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 31 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    cf.cond_br %early, ^exit, ^work
  ^exit:
    tt.return
  ^work:
    // CHECK: ^bb2:
    // CHECK-DAG: %[[A:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 30 : i32}
    // CHECK-DAG: %[[B:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 30 : i32}
    // CHECK-DAG: %[[TMEM:.*]], %[[TOKEN:.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 31 : i32}
    // CHECK: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      %view = ttg.memdesc_subslice %a[0] {async_task_id = array<i32: 2>} : !ttg.memdesc<2xi32, #shared1, #smem, mutable> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
      ttg.local_store %value, %view {async_task_id = array<i32: 2>} : tensor<1xi32, #blocked1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
      %loaded = ttg.local_load %b {async_task_id = array<i32: 2>} : !ttg.memdesc<2xi32, #shared1, #smem, mutable> -> tensor<2xi32, #blocked1>
      %true = arith.constant {async_task_id = array<i32: 2>} true
      %stored = ttng.tmem_store %accValue, %tmem[%token], %true {async_task_id = array<i32: 2>} : tensor<128x128xf32, #acc> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tmemValue, %tmemToken = ttng.tmem_load %tmem[%stored] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc>
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "gemm"],
       ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}
