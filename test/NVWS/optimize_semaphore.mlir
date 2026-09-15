// RUN: triton-opt %s --split-input-file --allow-unregistered-dialect --nvws-semaphore-optimize=num-stages=3 | FileCheck %s --implicit-check-not=ttng.init_barrier --implicit-check-not=ttng.wait_barrier

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @optimize_only
  tt.func @optimize_only(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // CHECK: [[A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>
    %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %empty_a = nvws.semaphore.create %a released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>
    %full_a = nvws.semaphore.create %a {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>
    %empty_b = nvws.semaphore.create %b released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>
    %full_b = nvws.semaphore.create %b {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] released = 7 {pending_count = 1 : i32}
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32}
    // CHECK-NOT: nvws.semaphore.create
    // CHECK: scf.for {{.*}} iter_args({{%.*}} = {{%.*}}) -> (!ttg.async.token)
    %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %token) -> (!ttg.async.token) : i32 {
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: {{%.*}}:2 = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %pa = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]> -> !ttg.async.token
      %pba = nvws.semaphore.buffer %empty_a, %pa {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
      nvws.semaphore.release %full_a, %pa [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token
      %pb = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]> -> !ttg.async.token
      %pbb = nvws.semaphore.buffer %empty_b, %pb {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
      nvws.semaphore.release %full_b, %pb [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
      %ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]> -> !ttg.async.token
      %cba = nvws.semaphore.buffer %full_a, %ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
      %cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]> -> !ttg.async.token
      %cbb = nvws.semaphore.buffer %full_b, %cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>
      // CHECK: [[RHS:%.*]] = ttg.memdesc_trans [[CBUF]]#1
      %rhs = ttg.memdesc_trans %cbb {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma [[CBUF]]#0, [[RHS]]
      %mma = ttng.tc_gen5_mma %cba, %rhs, %acc[%iter], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_a, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared_b, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_a, %ca [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %cb [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_a, #smem, mutable>]>, !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
    tt.return
  }
}

// -----

// Combining and multibuffering must retarget real producer and consumer users,
// including allocShape-bearing views, when the input ring already has depth 2.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @combine_depth_two_with_users
  tt.func @combine_depth_two_with_users(%desc_a: !tt.tensordesc<128x64xf16, #shared>, %desc_b: !tt.tensordesc<128x64xf16, #shared>, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // CHECK: [[A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %empty_a = nvws.semaphore.create %a released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %full_a = nvws.semaphore.create %a {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %empty_b = nvws.semaphore.create %b released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %full_b = nvws.semaphore.create %b {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] released = 7 {pending_count = 1 : i32}
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32}
    // CHECK-NOT: nvws.semaphore.create
    %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %token) -> (!ttg.async.token) : i32 {
      %pa = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pba = nvws.semaphore.buffer %empty_a, %pa {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.descriptor_load %desc_a[%iv, %iv] 16384 %pba {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.semaphore.release %full_a, %pa [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %pb = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbb = nvws.semaphore.buffer %empty_b, %pb {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.descriptor_load %desc_b[%iv, %iv] 16384 %pbb {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.semaphore.release %full_b, %pb [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
      // CHECK: [[PBUFS:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[PTOK]]
      // CHECK-DAG: nvws.descriptor_load %arg0{{.*}} [[PBUFS]]#0
      // CHECK-DAG: nvws.descriptor_load %arg1{{.*}} [[PBUFS]]#1
      %ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cba = nvws.semaphore.buffer %full_a, %ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      %cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbb = nvws.semaphore.buffer %full_b, %cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      %rhs = ttg.memdesc_trans %cbb {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64> -> !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable, 1x64x128>
      %mma = ttng.tc_gen5_mma %cba, %rhs, %acc[%iter], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable, 1x64x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]]
      // CHECK: [[CBUFS:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[CTOK]]
      // CHECK: ttg.memdesc_trans [[CBUFS]]#1
      // CHECK: ttng.tc_gen5_mma [[CBUFS]]#0
      nvws.semaphore.release %empty_a, %ca [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %cb [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
    tt.return
  }
}

// -----

// A single semaphore can own multiple backing buffers. Multibuffering must
// expand every backing allocation and preserve result order without combining.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @multiple_backing_buffers
  tt.func @multiple_backing_buffers(%desc_a: !tt.tensordesc<128x64xf16, #shared>, %desc_b: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK: [[A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create [[A]], [[B]] {pending_count = 1 : i32}
    %sem = nvws.semaphore.create %a, %b {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]]
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[VIEWS:%.*]]:2 = nvws.semaphore.buffer [[SEM]], [[TOK]]
    %views:2 = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
    // CHECK: nvws.descriptor_load %arg0{{.*}} [[VIEWS]]#0
    nvws.descriptor_load %desc_a[%c0, %c0] 4096 %views#0 : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
    // CHECK: nvws.descriptor_load %arg1{{.*}} [[VIEWS]]#1
    nvws.descriptor_load %desc_b[%c0, %c0] 4096 %views#1 : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Semaphore pairs with different physical depths cannot share one stage
// cursor. Their buffer.copy allocations also make them ineligible for
// multibuffering, so the four original semaphores must remain intact.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @do_not_combine_mixed_depth
  tt.func @do_not_combine_mixed_depth(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // CHECK: [[A:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32} : () -> !ttg.memdesc<3x128x64xf16
    %a = ttg.local_alloc {buffer.copy = 3 : i32} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[B:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x64xf16
    %b = ttg.local_alloc {buffer.copy = 2 : i32} : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-DAG: [[EA:%.*]] = nvws.semaphore.create [[A]] released = 7
    %empty_a = nvws.semaphore.create %a released = 7 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // CHECK-DAG: [[FA:%.*]] = nvws.semaphore.create [[A]] {pending_count = 1 : i32}
    %full_a = nvws.semaphore.create %a {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
    // CHECK-DAG: [[EB:%.*]] = nvws.semaphore.create [[B]] released = 3
    %empty_b = nvws.semaphore.create %b released = 3 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    // CHECK-DAG: [[FB:%.*]] = nvws.semaphore.create [[B]] {pending_count = 1 : i32}
    %full_b = nvws.semaphore.create %b {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %token) -> (!ttg.async.token) : i32 {
      %pa = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pba = nvws.semaphore.buffer %empty_a, %pa {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full_a, %pa [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %pb = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbb = nvws.semaphore.buffer %empty_b, %pb {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full_b, %pb [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.acquire [[FA]]
      %ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cba = nvws.semaphore.buffer %full_a, %ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.acquire [[FB]]
      %cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbb = nvws.semaphore.buffer %full_b, %cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %rhs = ttg.memdesc_trans %cbb {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>
      %mma = ttng.tc_gen5_mma %cba, %rhs, %acc[%iter], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      nvws.semaphore.release %empty_a, %ca [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %cb [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
    tt.return
  }
}

// -----

// Standalone TMA protocols are multibuffered even when they are not nested in
// a warp-specialized loop. Cover both descriptor-load producer forms carried
// by the tma_load async kind.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @standalone_tma_load
  tt.func @standalone_tma_load(%desc: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK: [[LOAD_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[LOAD_SEM:%.*]] = nvws.semaphore.create [[LOAD_BUF]] {pending_count = 1 : i32}
    %sem = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[LOAD_TOK:%.*]] = nvws.semaphore.acquire [[LOAD_SEM]]
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[LOAD_VIEW:%.*]] = nvws.semaphore.buffer [[LOAD_SEM]], [[LOAD_TOK]]
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: nvws.descriptor_load %arg0[%c0_i32, %c0_i32] 4096 [[LOAD_VIEW]]
    nvws.descriptor_load %desc[%c0, %c0] 4096 %view : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[LOAD_SEM]], [[LOAD_TOK]] [#nvws.async_op<tma_load>]
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @standalone_tma_gather
  tt.func @standalone_tma_gather(%desc: !tt.tensordesc<1x128xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    %offs = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    // CHECK: [[GATHER_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x8x128xf16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>
    // CHECK: [[GATHER_SEM:%.*]] = nvws.semaphore.create [[GATHER_BUF]] {pending_count = 1 : i32}
    %sem = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[GATHER_VIEW:%.*]] = nvws.semaphore.buffer [[GATHER_SEM]], %{{.*}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<8x128xf16, #shared, #smem, mutable>
    // CHECK: nvws.descriptor_gather %arg0[%cst, %c0_i32] 2048 [[GATHER_VIEW]]
    nvws.descriptor_gather %desc[%offs, %c0] 2048 %view : !tt.tensordesc<1x128xf16, #shared>, tensor<8xi32>, i32, !ttg.memdesc<8x128xf16, #shared, #smem, mutable>
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x8x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// An authored loop depth overrides the pass default. Zero disables expansion;
// depth two expands the whole semaphore pair, its released mask, and preserves
// a non-unit authored pending count.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @num_stages_zero
  tt.func @num_stages_zero(%desc: !tt.tensordesc<128x64xf16, #shared>, %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    // CHECK: [[ZERO_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[ZERO_EMPTY:%.*]] = nvws.semaphore.create [[ZERO_BUF]] released = 1
    %empty = nvws.semaphore.create %buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    scf.for %iv = %lb to %ub step %step : i32 {
      %tok = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %view = nvws.semaphore.buffer %empty, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%iv, %c0] 4096 %view {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full, %tok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.num_stages = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.outputs = []}
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @num_stages_two
  tt.func @num_stages_two(%desc: !tt.tensordesc<128x64xf16, #shared>, %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    // CHECK: [[TWO_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[TWO_EMPTY:%.*]] = nvws.semaphore.create [[TWO_BUF]] released = 3 {pending_count = 2 : i32}
    %empty = nvws.semaphore.create %buf released = 1 {pending_count = 2 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[TWO_FULL:%.*]] = nvws.semaphore.create [[TWO_BUF]] {pending_count = 1 : i32}
    %full = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    scf.for %iv = %lb to %ub step %step : i32 {
      %ptok = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %empty, %ptok {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%iv, %c0] 4096 %pbuf {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full, %ptok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %ctok0 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbuf0 = nvws.semaphore.buffer %full, %ctok0 {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %val0 = ttg.local_load %cbuf0 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      nvws.semaphore.release %empty, %ctok0 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %ctok1 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbuf1 = nvws.semaphore.buffer %full, %ctok1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %val1 = ttg.local_load %cbuf1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      nvws.semaphore.release %empty, %ctok1 [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%val0, %val1) : (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = []}
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A TMA-backed shared-memory group expands while unrelated non-TMA shared and
// tensor-memory semaphore groups remain at depth one.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @selective_multibuffer
  tt.func @selective_multibuffer() {
    // CHECK: [[TMA_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32
    %tma_buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[TMA_SEM:%.*]] = nvws.semaphore.create [[TMA_BUF]] released = 7
    %tma_sem = nvws.semaphore.create %tma_buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %tma_tok = nvws.semaphore.acquire %tma_sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %tma_sem, %tma_tok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    // CHECK: [[SYNC_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32
    %sync_buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[SYNC_SEM:%.*]] = nvws.semaphore.create [[SYNC_BUF]] released = 1
    %sync_sem = nvws.semaphore.create %sync_buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %sync_tok = nvws.semaphore.acquire %sync_sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sync_sem, %sync_tok [#nvws.async_op<none>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    %tm = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: nvws.semaphore.create %{{.*}} released = 1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf32
    %tm_sem = nvws.semaphore.create %tm released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %tm_tok = nvws.semaphore.acquire %tm_sem : !nvws.semaphore<[!ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %tm_sem, %tm_tok [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @optimize_mixed_completion_kinds
  tt.func @optimize_mixed_completion_kinds(%desc: !tt.tensordesc<128x64xf16, #shared>, %src: tensor<128x64xf16, #blocked>, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Every input semaphore correctly authors pending_count = 1.
    %empty_a = nvws.semaphore.create %a released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %full_a = nvws.semaphore.create %a {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %empty_b = nvws.semaphore.create %b released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %full_b = nvws.semaphore.create %b {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // Duplicate tc5mma kinds collapse to one EMPTY completion.
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create {{%.*}}, {{%.*}} released = 7 {pending_count = 1 : i32}
    // Distinct tma_load and none kinds require two FULL completions.
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create {{%.*}}, {{%.*}} {pending_count = 2 : i32}
    %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %token) -> (!ttg.async.token) : i32 {
      %pa = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pba = nvws.semaphore.buffer %empty_a, %pa {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%iv, %iv] 16384 %pba {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full_a, %pa [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %pb = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbb = nvws.semaphore.buffer %empty_b, %pb {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %src, %pbb {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], {{%.*}} [#nvws.async_op<tma_load>, #nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      nvws.semaphore.release %full_b, %pb [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cba = nvws.semaphore.buffer %full_a, %ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbb = nvws.semaphore.buffer %full_b, %cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %rhs = ttg.memdesc_trans %cbb {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>
      %mma = ttng.tc_gen5_mma %cba, %rhs, %acc[%iter], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]], {{%.*}} [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_a, %ca [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %cb [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
    tt.return
  }
}
