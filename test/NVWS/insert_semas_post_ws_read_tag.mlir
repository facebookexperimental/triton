// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore --tritongpu-partition-loops | FileCheck %s --check-prefix=PARTITION

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // LOWER-LABEL: @post_ws_tmem_read_tag
  // LOWER: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // LOWER: ttng.wait_barrier {{.*}} :
  // LOWER: [[READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
  // LOWER: [[OUT:%.*]], {{%.*}} = ttng.tmem_load [[READ]][]
  // LOWER-NEXT: "use"([[OUT]])
  // PARTITION-LABEL: @post_ws_tmem_read_tag
  // PARTITION: nvws.warp_group
  // PARTITION: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // PARTITION: nvws.warp_group.return
  // PARTITION: [[POST_READ:%.*]] = ttg.memdesc_index {{.*}} : !ttg.memdesc<1x128x128xf32
  // PARTITION: [[POST_OUT:%.*]], {{%.*}} = ttng.tmem_load [[POST_READ]][]
  // PARTITION-NEXT: "use"([[POST_OUT]])
  // CHECK-LABEL: @post_ws_tmem_read_tag
  tt.func @post_ws_tmem_read_tag(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32}
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 1 : i32}
    // CHECK-NEXT: [[HELD:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[INIT_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]]
    // CHECK-NEXT: ttng.tmem_store %{{.*}}, [[INIT_BUF]][], %{{.*}}
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // The same semaphore token remains live across the loop. Every MMA uses
    // its buffer, and one release after the loop tracks the final MMA.
    // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT: [[BODY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[HELD]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[BODY_BUF]][], %{{.*}}, %{{.*}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[HELD]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[READ]]
    // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[READ_BUF]][]
    %out, %load_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // The post-loop read is the last access; no release of [[EMPTY]] follows it.
    // CHECK-NOT: nvws.semaphore.release
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
