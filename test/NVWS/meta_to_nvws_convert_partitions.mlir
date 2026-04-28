// RUN: triton-opt %s -allow-unregistered-dialect --nvws-meta-to-nvws-convert | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-meta-to-nvws-convert --nvws-meta-to-nvws-convert | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @partitions_and_result
  tt.func @partitions_and_result(%lb: i32, %ub: i32, %step: i32,
                                 %init: tensor<32xf32, #blocked>) {
    // CHECK: %[[LOOP:.*]] = scf.for
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%arg = %init) -> (tensor<32xf32, #blocked>) : i32 {
      // CHECK: "test.producer"() {ttg.partition = array<i32: 1>}
      %next = "test.producer"() {async_task_id = array<i32: 1>} : () -> tensor<32xf32, #blocked>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} {{.*}}
      scf.yield {async_task_id = array<i32: 0, 1>} %next : tensor<32xf32, #blocked>
    // The exact attribute dictionary verifies that async_task_id was consumed.
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "compute", "load"], ttg.warp_specialize.tag = 7 : i32}
    } {async_task_id = array<i32: 2, 0, 1, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "compute", "load"],
       ttg.warp_specialize.tag = 7 : i32}
    // CHECK: "test.outside"(%[[LOOP]]) {async_task_id = array<i32: 0>} :
    "test.outside"(%result) {async_task_id = array<i32: 0>, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 7 : i32} : (tensor<32xf32, #blocked>) -> ()
    tt.return
  }

  // A yielded producer outside the WS loop remains in Meta representation
  // until conversion. Its source attribute is consumed into partition.outputs
  // without adding ttg.partition to the external constant.
  // CHECK-LABEL: tt.func @external_yielded_meta_producer
  tt.func @external_yielded_meta_producer(
      %lb: i32, %ub: i32, %step: i32, %init: i1) {
    // CHECK: %[[TRUE:.*]] = arith.constant true
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    // CHECK: %[[BOOL_LOOP:.*]] = scf.for
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%flag = %init) -> (i1) : i32 {
      // CHECK: "test.bool.consumer"({{.*}}) {ttg.partition = array<i32: 1>}
      "test.bool.consumer"(%flag) {async_task_id = array<i32: 1>} : (i1) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} %[[TRUE]]
      scf.yield {async_task_id = array<i32: 0>} %true : i1
    // The exact attribute dictionary verifies that async_task_id was consumed.
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["default", "compute"], ttg.warp_specialize.tag = 9 : i32}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["default", "compute"],
       ttg.warp_specialize.tag = 9 : i32}
    // CHECK: "test.bool.outside"(%[[BOOL_LOOP]]) {async_task_id = array<i32: 0>}
    "test.bool.outside"(%result) {async_task_id = array<i32: 0>} : (i1) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @token_result
  tt.func @token_result(%lb: i32, %ub: i32, %step: i32,
                        %init: !ttg.async.token) {
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%arg = %init) -> (!ttg.async.token) : i32 {
      // CHECK: "test.token.consumer"({{.*}}) {ttg.partition = array<i32: 1>}
      "test.token.consumer"(%arg) {async_task_id = array<i32: 1>} : (!ttg.async.token) -> ()
      // CHECK: %[[NEXT:.*]] = "test.token.producer"() {ttg.partition = array<i32: 0>}
      %next = "test.token.producer"() {async_task_id = array<i32: 0>} : () -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %[[NEXT]]
      scf.yield {async_task_id = array<i32: 0, 1>} %next : !ttg.async.token
    // The exact attribute dictionary verifies that async_task_id was consumed.
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["producer", "consumer"], ttg.warp_specialize.tag = 8 : i32}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["producer", "consumer"],
       ttg.warp_specialize.tag = 8 : i32}
    // CHECK: "test.token.outside"({{.*}}) {async_task_id = array<i32: 0>}
    "test.token.outside"(%result) {async_task_id = array<i32: 0>} : (!ttg.async.token) -> ()
    tt.return
  }

  // An alias outside the WS loop is replayed inside it by InsertSemas, so its
  // Meta task assignment must become persistent NVWS ownership. Running the
  // converter twice must retain the same partition and tag.
  // CHECK-LABEL: tt.func @external_memdesc_alias
  tt.func @external_memdesc_alias(
      %lb: i32, %ub: i32, %step: i32) {
    %buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 90 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %[[VIEW:.*]] = ttg.memdesc_reinterpret %{{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 10 : i32}
    %view = ttg.memdesc_reinterpret %buffer {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %{{.*}} {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 10 : i32}
    %trans = ttg.memdesc_trans %view {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: "test.memdesc.consumer"(%[[TRANS]]) {ttg.partition = array<i32: 1>}
      "test.memdesc.consumer"(%trans) {async_task_id = array<i32: 1>} : (!ttg.memdesc<64x128xf16, #shared_t, #smem, mutable>) -> ()
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["default", "compute"],
       ttg.warp_specialize.tag = 10 : i32}
    tt.return
  }

  // Meta schedules the inner loop but specializes the enclosing task-bearing
  // loop nest. The converter promotes only the WS-root metadata; the inner
  // loop keeps its authored pipeline schedule. The file's second RUN verifies
  // that repeating conversion preserves this shape.
  // CHECK-LABEL: tt.func @promote_nested_meta_ws_root
  tt.func @promote_nested_meta_ws_root(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: scf.for
    scf.for %outer = %lb to %ub step %step : i32 {
      "test.outer.before"() {async_task_id = array<i32: 1>} : () -> ()
      // CHECK: scf.for
      scf.for %inner = %lb to %ub step %step : i32 {
        "test.inner"() {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : () -> ()
        scf.yield {async_task_id = array<i32: 0, 1>}
      // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
      } {async_task_id = array<i32: 0, 1>, tt.num_stages = 2 : i32,
         tt.scheduled_max_stage = 1 : i32, tt.warp_specialize,
         ttg.partition.stages = [0 : i32, 1 : i32],
         ttg.partition.types = ["default", "gemm"],
         ttg.warp_specialize.tag = 11 : i32}
      "test.outer.after"() {async_task_id = array<i32: 0>} : () -> ()
      scf.yield {async_task_id = array<i32: 0, 1>}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.partition.types = ["default", "gemm"], ttg.warp_specialize.tag = 11 : i32}
    } {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
