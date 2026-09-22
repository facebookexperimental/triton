// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-meta-to-nvws-convert \
// RUN:   | FileCheck %s --implicit-check-not='{{ttg\.memdesc_reinterpret|allocation\.reuseTarget|buffer\.id = 22}}'
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-meta-to-nvws-convert --nvws-meta-to-nvws-convert \
// RUN:   | FileCheck %s --implicit-check-not='{{ttg\.memdesc_reinterpret|allocation\.reuseTarget|buffer\.id = 22}}'

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
  // loop nest. The converter promotes the WS-root metadata and its effective
  // SMEM policy, overriding the outer policy; the inner loop keeps its pipeline
  // schedule. The file's second RUN verifies that repeating conversion preserves
  // both the promoted policy and the circular buffer plan.
  // CHECK-LABEL: tt.func @promote_nested_meta_ws_root
  tt.func @promote_nested_meta_ws_root(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[A:.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 70 : i32, buffer.start = 0 : i32}
    %a = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 70 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 70 : i32, buffer.start = 1 : i32}
    %b = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 70 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: scf.for
    scf.for %outer = %lb to %ub step %step : i32 {
      "test.outer.before"() {async_task_id = array<i32: 1>} : () -> ()
      // CHECK: scf.for
      scf.for %inner = %lb to %ub step %step : i32 {
        "test.inner"(%a, %b) {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : (!ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
        scf.yield {async_task_id = array<i32: 0, 1>}
      // CHECK: } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
      } {async_task_id = array<i32: 0, 1>, tt.num_stages = 2 : i32,
         tt.scheduled_max_stage = 1 : i32, tt.warp_specialize,
         tt.smem_alloc_algo = 0 : i32, tt.smem_circular_reuse = true,
         ttg.partition.stages = [0 : i32, 1 : i32],
         ttg.partition.types = ["default", "gemm"],
         ttg.warp_specialize.tag = 11 : i32}
      "test.outer.after"() {async_task_id = array<i32: 0>} : () -> ()
      scf.yield {async_task_id = array<i32: 0, 1>}
    // CHECK: } {tt.smem_alloc_algo = 0 : i32, tt.smem_circular_reuse = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.partition.types = ["default", "gemm"], ttg.warp_specialize.tag = 11 : i32}
    } {async_task_id = array<i32: 0, 1>, tt.smem_alloc_algo = 1 : i32,
       tt.smem_circular_reuse = false}
    tt.return
  }
}

// -----
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
      %view = ttg.memdesc_subslice %a[0] {async_task_id = array<i32: 2>} : !ttg.memdesc<2xi32, #shared1, #smem, mutable> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable, 2>
      ttg.local_store %value, %view {async_task_id = array<i32: 2>} : tensor<1xi32, #blocked1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable, 2>
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

// -----
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @descriptor_to_planned_buffer
  tt.func @descriptor_to_planned_buffer(
      %load_desc: !tt.tensordesc<64x64xf16, #shared>,
      %gather_desc: !tt.tensordesc<1x64xf16, #shared>,
      %offsets: tensor<128xi32, #offsets>, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[LOAD_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
    %load_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[GATHER_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}
    %gather_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %[[PREHEADER_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32}
    %preheader_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // Descriptor operations are already in NVWS form when they reach the bridge.
    // CHECK-NOT: tt.descriptor_load
    // CHECK: nvws.descriptor_load {{.*}} 8192 %[[PREHEADER_BUFFER]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
    nvws.descriptor_load %load_desc[%lb, %lb] 8192 %preheader_buffer {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x64xf16, #shared>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-NOT: tt.descriptor_gather
    // CHECK: nvws.descriptor_gather {{.*}} 16384 %[[GATHER_BUFFER]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
    nvws.descriptor_gather %gather_desc[%offsets, %lb] 16384 %gather_buffer {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x64xf16, #shared>, tensor<128xi32, #offsets>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: nvws.descriptor_load {{.*}} 8192 %[[LOAD_BUFFER]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      nvws.descriptor_load %load_desc[%i, %i] 8192 %load_buffer {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x64xf16, #shared>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK-NOT: async_task_id
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "load"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @direct_tma_reduce_wait
  tt.func @direct_tma_reduce_wait(
      %desc: !tt.tensordesc<128x128xf32, #shared>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    scf.for %i = %lb to %ub step %step : i32 {
      %staging = ttg.local_alloc {async_task_id = array<i32: 0>} :
          () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>

      // CHECK: %[[TOKEN:.*]] = ttng.async_tma_reduce add
      // CHECK-SAME: {ttg.partition = array<i32: 0>}
      %token = ttng.async_tma_reduce add, %desc[%c0, %c0] %staging
          {async_task_id = array<i32: 0>} :
          !tt.tensordesc<128x128xf32, #shared>,
          !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
          -> !ttg.async.token

      // The Meta schedule places this wait in an epilogue partition. The NVWS
      // conversion must co-locate it with its direct token producer.
      // CHECK: ttng.async_tma_store_token_wait %[[TOKEN]]
      // CHECK-SAME: {can_rotate_by_buffer_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttng.async_tma_store_token_wait %token
          {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} :
          !ttg.async.token
      scf.yield {async_task_id = array<i32: 0, 4>}
    } {async_task_id = array<i32: 0, 4>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["reduction", "unused", "unused", "unused", "epilogue"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
