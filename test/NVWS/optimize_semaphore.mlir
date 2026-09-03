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
