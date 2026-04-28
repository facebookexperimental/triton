// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase --cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @circular_model_descriptor_load_mma
  tt.func @circular_model_descriptor_load_mma(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 3>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 300 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_STEP:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 1 : i32
      // CHECK: [[V_NEXT:%.*]] = arith.addi [[K_STAGE]], [[V_STEP]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_DEPTH:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 2 : i32
      // CHECK: [[V_WRAP:%.*]] = arith.cmpi eq, [[V_NEXT]], [[V_DEPTH]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_ZERO:%.*]] = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select [[V_WRAP]], [[V_ZERO]], [[V_NEXT]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1_k = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1_k], %tk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1_k], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v_use = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
  tt.func @circular_tutorial_1_1_to_2_2(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 301 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
  tt.func @circular_tutorial_1_2_to_3_4(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 302 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3, 4>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3, 4>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 4>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 4>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
  tt.func @circular_tutorial_1_1_to_2_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 303 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z0], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z0] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z0], %tv {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z0], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
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
  tt.func @circular_tutorial_1_2_to_3_3(%lb: i32, %ub: i32, %step: i32) {
    %payload_k = "make_k"() {ttg.partition = array<i32: 1>} : () -> tensor<128x128xf16, #blocked>
    %payload_v = "make_v"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 304 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE:%.*]] true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z_k = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      // CHECK: [[K_STEP:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 1 : i32
      // CHECK: [[K_NEXT:%.*]] = arith.addi {{%.*}}, [[K_STEP]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 2 : i32
      // CHECK: [[K_WRAP:%.*]] = arith.cmpi eq, [[K_NEXT]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} 0 : i32
      // CHECK: [[K_STAGE:%.*]] = arith.select [[K_WRAP]], [[K_ZERO]], [[K_NEXT]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[K_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[K_STAGE]], [[K_PHASE]]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[K_STAGE]]], [[K_EMPTY_TOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[K_STAGE]]], [[K_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk = nvws.semaphore.acquire %empty[%z_k] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk = nvws.semaphore.buffer %empty[%z_k], %tk {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_k, %bk {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_k], %tk [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z_v = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      // CHECK: [[V_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[V_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[V_EMPTY_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[V_STAGE]], [[V_PHASE]]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[V_STAGE]]], [[V_EMPTY_TOK]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]][[[V_STAGE]]], [[V_EMPTY_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv = nvws.semaphore.acquire %empty[%z_v] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv = nvws.semaphore.buffer %empty[%z_v], %tv {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload_v, %bv {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full[%z_v], %tv [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %m1 = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_ACQ_M1:%.*]] = arith.constant {ttg.partition = array<i32: 1, 2, 3>} -1 : i32
      // CHECK: [[K_ACQ_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_ACQ_M1]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_REM:%.*]] = arith.remsi [[K_ACQ_RAW]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_NEG:%.*]] = arith.cmpi slt, [[K_ACQ_REM]], [[K_ZERO]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_WRAP:%.*]] = arith.addi [[K_ACQ_REM]], [[K_DEPTH]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_ACQ_STAGE:%.*]] = arith.select [[K_ACQ_NEG]], [[K_ACQ_WRAP]], [[K_ACQ_REM]] {ttg.partition = array<i32: 1, 2, 3>} : i32
      // CHECK: [[K_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[K_ACQ_STAGE]], [[K_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[K_BUF_M1:%.*]] = arith.constant {ttg.partition = array<i32: 3>} -1 : i32
      // CHECK: [[K_BUF_RAW:%.*]] = arith.addi [[V_STAGE]], [[K_BUF_M1]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_DEPTH:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 2 : i32
      // CHECK: [[K_BUF_REM:%.*]] = arith.remsi [[K_BUF_RAW]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[K_BUF_NEG:%.*]] = arith.cmpi slt, [[K_BUF_REM]], [[K_BUF_ZERO]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_WRAP:%.*]] = arith.addi [[K_BUF_REM]], [[K_BUF_DEPTH]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[K_BUF_STAGE:%.*]] = arith.select [[K_BUF_NEG]], [[K_BUF_WRAP]], [[K_BUF_REM]] {ttg.partition = array<i32: 3>} : i32
      // CHECK: nvws.semaphore.buffer [[FULL]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[K_BUF_STAGE]]], [[K_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tk_use = nvws.semaphore.acquire %full[%m1] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bk_use = nvws.semaphore.buffer %full[%m1], %tk_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kval = ttg.local_load %bk_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%m1], %tk_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token

      %z = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      // CHECK: [[V_FULL_PHASE:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 3>} : i32
      // CHECK: [[V_FULL_TOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[V_STAGE]], [[V_FULL_PHASE]]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.buffer [[FULL]][[[V_STAGE]]], [[V_FULL_TOK]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[EMPTY]][[[V_STAGE]]], [[V_FULL_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %tv_use = nvws.semaphore.acquire %full[%z] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %bv_use = nvws.semaphore.buffer %full[%z], %tv_use {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vval = ttg.local_load %bv_use {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      nvws.semaphore.release %empty[%z], %tv_use [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%kval, %vval) {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked>, tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }
}
