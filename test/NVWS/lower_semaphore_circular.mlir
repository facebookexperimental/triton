// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-semaphore --cse | FileCheck %s --implicit-check-not=nvws.semaphore

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: [[FULL_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  tt.func @circular_model_descriptor_load_mma(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[K_EMPTY_WAIT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: [[K_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttg.local_store {{.*}}, [[K_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: [[K_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[K_FULL_REL]], 1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      %tk = nvws.semaphore.acquire %empty[%z_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[V_EMPTY_WAIT]], {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: [[V_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttg.local_store {{.*}}, [[V_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: [[V_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[V_FULL_REL]], 1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
      %tv = nvws.semaphore.acquire %empty[%z_v] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_USE_STAGE:%.*]] = arith.select {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_USE_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[K_FULL_WAIT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[K_USE_BUF_STAGE:%.*]] = arith.select {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_USE_BUF_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_USE_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[K_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_USE_BUF_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[K_EMPTY_REL]], 1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      %tk_use = nvws.semaphore.acquire %full[%m1_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1_k], %tk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1_k], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v_use = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[V_FULL_WAIT]], {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[V_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_USE_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[V_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[V_EMPTY_REL]], 1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 1>}
      %tv_use = nvws.semaphore.acquire %full[%z_v_use] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z_v_use], %tv_use {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z_v_use], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_2
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: [[FULL_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  tt.func @circular_tutorial_1_1_to_2_2(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[K_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[K_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[K_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[K_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[K_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[V_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[V_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[V_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[V_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[V_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[V_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_USE_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_USE_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[K_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_USE_BUF]] {ttg.partition = array<i32: 2>}
      // CHECK: [[K_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[K_EMPTY_REL]], 1 {ttg.partition = array<i32: 2>}
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[V_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[V_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_USE_BUF]] {ttg.partition = array<i32: 2>}
      // CHECK: [[V_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[V_EMPTY_REL]], 1 {ttg.partition = array<i32: 2>}
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2>}
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
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: [[FULL_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  tt.func @circular_tutorial_1_2_to_3_4(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[K_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[K_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[K_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[K_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[K_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[V_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[V_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[V_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_store {{.*}}, [[V_BUF]] {ttg.partition = array<i32: 2>}
      // CHECK: [[V_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[V_FULL_REL]], 1 {ttg.partition = array<i32: 2>}
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_USE_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_USE_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[K_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 3>}
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_USE_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK: [[K_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[K_EMPTY_REL]], 1 {ttg.partition = array<i32: 3>}
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // CHECK: [[V_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 4>}
      // CHECK: ttng.wait_barrier [[V_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 4>}
      // CHECK: [[V_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 4>}
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_USE_BUF]] {ttg.partition = array<i32: 4>}
      // CHECK: [[V_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 4>}
      // CHECK: ttng.arrive_barrier [[V_EMPTY_REL]], 1 {ttg.partition = array<i32: 4>}
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3, 4>}
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
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: [[FULL_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  tt.func @circular_tutorial_1_1_to_2_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[K_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[K_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[K_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[K_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[K_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[V_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[V_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[V_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[V_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[V_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_USE_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_USE_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[K_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_USE_BUF]] {ttg.partition = array<i32: 2>}
      // CHECK: [[K_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[K_EMPTY_REL]], 1 {ttg.partition = array<i32: 2>}
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[V_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 3>}
      // CHECK: [[V_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_USE_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK: [[V_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[V_EMPTY_REL]], 1 {ttg.partition = array<i32: 3>}
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2, 3>}
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
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: [[FULL_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttng.init_barrier {{%.*}}, 1
  // CHECK: ttng.init_barrier {{%.*}}, 1
  tt.func @circular_tutorial_1_2_to_3_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[K_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[K_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store {{.*}}, [[K_BUF]] {ttg.partition = array<i32: 1>}
      // CHECK: [[K_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[K_FULL_REL]], 1 {ttg.partition = array<i32: 1>}
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[V_EMPTY_WAIT]], {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[V_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_store {{.*}}, [[V_BUF]] {ttg.partition = array<i32: 2>}
      // CHECK: [[V_FULL_REL:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[V_FULL_REL]], 1 {ttg.partition = array<i32: 2>}
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_USE_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[K_USE_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[K_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 3>}
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select {{.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_USE_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK: [[K_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[K_BUF_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[K_EMPTY_REL]], 1 {ttg.partition = array<i32: 3>}
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_WAIT:%.*]] = ttg.memdesc_index [[FULL_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.wait_barrier [[V_FULL_WAIT]], {{%.*}} {ttg.partition = array<i32: 3>}
      // CHECK: [[V_USE_BUF:%.*]] = ttg.memdesc_index [[BASE]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_USE_BUF]] {ttg.partition = array<i32: 3>}
      // CHECK: [[V_EMPTY_REL:%.*]] = ttg.memdesc_index [[EMPTY_MBAR]][[[V_STAGE]]] {ttg.partition = array<i32: 3>}
      // CHECK: ttng.arrive_barrier [[V_EMPTY_REL]], 1 {ttg.partition = array<i32: 3>}
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3>}
      %tv_use = nvws.semaphore.acquire %full[%z] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z], %tv_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }
}
