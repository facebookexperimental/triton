// RUN: triton-opt %s --nvws-memory-planner=num-buffers=3 -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @loop_carried_use_acc_false_init
  tt.func public @loop_carried_use_acc_false_init(%a: !ttg.memdesc<64x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared_t, #smem>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %false = arith.constant false
    %true = arith.constant true
    %zero = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>

    // CHECK: ttng.tmem_alloc %{{.*}} {buffer.copy = {{[0-9]+}} : i32, buffer.id = 0 : i32, buffer.offset = 0 : i32
    %acc, %tok = ttng.tmem_alloc %zero {ttg.partition = array<i32: 2>} : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: scf.for
    // CHECK: ttng.tc_gen5_mma {{.*}} {tmem.start = array<i32: 0>, ttg.partition = array<i32: 2>}
    %loop:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%use_acc = %false, %acc_tok = %tok) -> (i1, !ttg.async.token) : i32 {
      %mma_tok = ttng.tc_gen5_mma %a, %b, %acc[%acc_tok], %use_acc, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_t, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 2>} %true, %mma_tok : i1, !ttg.async.token
    } {ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}

    // CHECK: ttng.tmem_load {{.*}} {tmem.end = array<i32: 0>, ttg.partition = array<i32: 0>}
    %out, %out_tok = ttng.tmem_load %acc[%loop#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
    "use"(%out) : (tensor<64x128xf32, #linear>) -> ()
    tt.return
  }

  // Meta's direct-user rule gives a sourceful TMEM operand A copies when its
  // consuming MMA has a loop-carried accumulator token. The wider D allocation
  // reaches copy 3 first; the remaining TMEM capacity raises A to copy 2.
  // CHECK-LABEL: @operand_a_follows_loop_carried_mma
  tt.func public @operand_a_follows_loop_carried_mma(%b: !ttg.memdesc<128x128xf16, #shared_t, #smem>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %false = arith.constant false
    %true = arith.constant true
    %a_init = arith.constant dense<1.000000e+00> : tensor<64x128xf16, #linear>
    %acc_init = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>

    // CHECK: ttng.tmem_alloc {{.*}}buffer.copy = 2 : i32, buffer.id = [[A_ID:[0-9]+]] : i32, buffer.offset = 0 : i32
    %a_tmem = ttng.tmem_alloc %a_init {ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #linear>) -> !ttg.memdesc<64x128xf16, #tmem, #ttng.tensor_memory>
    // CHECK: ttng.tmem_alloc {{.*}}buffer.copy = 3 : i32, buffer.id = [[D_ID:[0-9]+]] : i32, buffer.offset = 0 : i32
    %acc, %tok = ttng.tmem_alloc %acc_init {ttg.partition = array<i32: 2>} : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    %loop:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%use_acc = %false, %acc_tok = %tok) -> (i1, !ttg.async.token) : i32 {
      %mma_tok = ttng.tc_gen5_mma %a_tmem, %b, %acc[%acc_tok], %use_acc, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<64x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared_t, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 2>} %true, %mma_tok : i1, !ttg.async.token
    } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}

    %out, %out_tok = ttng.tmem_load %acc[%loop#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
    "use"(%out) : (tensor<64x128xf32, #linear>) -> ()
    tt.return
  }
}
