// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=COUNT

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  // COUNT-LABEL: @circular_model_descriptor_load_mma
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_model_descriptor_load_mma(
      %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>,
      %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>,
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[K_EMPTY_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_k[%c0, %c0] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kt = ttg.memdesc_trans %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[V_EMPTY_BUF]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_v[%c0, %c0] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_TRANS:%.*]] = ttg.memdesc_trans [[K_FULL_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
      // CHECK: [[QK:%.*]] = ttng.tc_gen5_mma {{.*}}, [[K_TRANS]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %qk = ttng.tc_gen5_mma %lhs, %kt, %acc[%tok], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[PV:%.*]] = ttng.tc_gen5_mma {{.*}}, [[V_FULL_BUF]], {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %pv = ttng.tc_gen5_mma %lhs, %v, %acc[%qk], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      "use_token"(%pv) {ttg.partition = array<i32: 1>} : (!ttg.async.token) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_2
  // COUNT-LABEL: @circular_tutorial_1_1_to_2_2
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_1_to_2_2(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 2>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_4
  // COUNT-LABEL: @circular_tutorial_1_2_to_3_4
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_2_to_3_4(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3, 4>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 3, 4>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3, 4>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4>], ttg.warp_specialize.tag = 2 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_1_to_2_3
  // COUNT-LABEL: @circular_tutorial_1_1_to_2_3
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_1_to_2_3(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 2, 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 2, 3>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_tutorial_1_2_to_3_3
  // COUNT-LABEL: @circular_tutorial_1_2_to_3_3
  // COUNT-COUNT-1: ttg.local_alloc
  // COUNT-NOT: ttg.local_alloc
  // COUNT-COUNT-2: nvws.semaphore.create
  // COUNT-NOT: ttg.local_alloc
  // COUNT-NOT: nvws.semaphore.create
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false {pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
  tt.func @circular_tutorial_1_2_to_3_3(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> !ty
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> !ty
    %k = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[K_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[K_EMPTY_BUF]] {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[K_EMPTY_STAGE]]{{\]}}, [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_k, %k {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[V_EMPTY_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}} {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_EMPTY_BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store {{.*}}, [[V_EMPTY_BUF]] {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]]{{\[}}[[V_EMPTY_STAGE]]{{\]}}, [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      ttg.local_store %payload_v, %v {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[K_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[KVAL:%.*]] = ttg.local_load [[K_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[K_FULL_STAGE]]{{\]}}, [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %kval = ttg.local_load %k {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: [[V_FULL_STAGE:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}} {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V_FULL_BUF:%.*]] = nvws.semaphore.buffer [[FULL]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[VVAL:%.*]] = ttg.local_load [[V_FULL_BUF]] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      // CHECK: nvws.semaphore.release [[EMPTY]]{{\[}}[[V_FULL_STAGE]]{{\]}}, [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %vval = ttg.local_load %v {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      // CHECK: "use"([[KVAL]], [[VVAL]]) {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
      "use"(%kval, %vval) {ttg.partition = array<i32: 3>} : (!ty, !ty) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
