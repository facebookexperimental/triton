// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-allocas | FileCheck %s --check-prefix=SMEM --implicit-check-not=nvws.semaphore.
// RUN: env NVWS_USE_SSA_TMEM=1 triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-allocas | FileCheck %s --check-prefix=TMEM --implicit-check-not=nvws.semaphore.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [128, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SMEM-LABEL: tt.func @warp_specialize_tma_matmul
  // SMEM: [[LHS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // SMEM: [[RHS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // SMEM: nvws.descriptor_load {{.*}} [[LHS]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // SMEM: [[RHS_CONSUMER:%.*]] = ttg.memdesc_trans [[RHS]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Two cross-partition values => two semaphore pairs
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // Producer LHS: acquire EMPTY1, buffer, descriptor_load, release FULL1
      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // Producer RHS: acquire EMPTY2, buffer, descriptor_load, release FULL2
      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>

      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // Consumer RHS: acquire FULL2, buffer, memdesc_trans uses buffer
      // Consumer LHS: acquire FULL1, buffer, MMA uses both buffers
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Cross-release: consumer releases EMPTY semaphores
      scf.yield {ttg.partition = array<i32: 0, 1>} %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @specialize_load_only(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 1 : i32, loop.stage = 0, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // Consumer: acquire FULL, buffer, local_load, release EMPTY
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // SMEM-LABEL: tt.func @no_value_semaphore
  // SMEM-NOT: ttg.local_alloc
  // SMEM: tt.return
  tt.func @no_value_semaphore(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // SMEM-LABEL: tt.func @value_semaphore_multiple_producers
  // SMEM: [[SSA_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared{{[0-9]*}}, #smem, mutable>
  // SMEM: ttg.local_store {{.*}}, [[SSA_BUF]]
  // SMEM: [[SSA_LOAD:%.*]] = ttg.local_load [[SSA_BUF]] {{.*}}ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared{{[0-9]*}}, #smem, mutable> -> tensor<128x64xf16, #blocked1>
  // SMEM: "use2"([[SSA_LOAD]])
  tt.func @value_semaphore_multiple_producers(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, store val, release FULL
      // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use0"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use1"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // SMEM-LABEL: tt.func @load_used_as_reg_and_smem
  // SMEM: [[REG_SMEM_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // SMEM: nvws.descriptor_load {{.*}} [[REG_SMEM_PRODUCER:%.*]] {{.*}}ttg.partition = array<i32: 2>}
  // SMEM: [[REG_LOAD:%.*]] = ttg.local_load {{.*}} {{.*}}ttg.partition = array<i32: 0>}
  // SMEM: "use1"([[REG_LOAD]]) {{.*}}ttg.partition = array<i32: 0>}
  // SMEM: "use2"([[REG_SMEM_BUF]]) {{.*}}ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Consumer 1 (register use): acquire FULL, buffer, local_load, release EMPTY
      // Consumer 2 (smem use): acquire FULL, buffer used directly, release EMPTY
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Single consumer partition 0: acquire FULL, buffer, local_load + uses, release EMPTY
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg4: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg5: !tt.tensordesc<tensor<128x8xi8, #shared2>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %result = ttng.tmem_alloc %cst_0 : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %0 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %1 = arith.muli %arg6, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %2 = tt.descriptor_load %arg3[%arg1, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %3 = tt.descriptor_load %arg4[%arg2, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = ttg.local_alloc %2 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      %6 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      // scales are a register descriptor_load — stays as tt.descriptor_load
      %4 = tt.descriptor_load %arg5[%arg1, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x8xi8, #shared2>> -> tensor<128x8xi8, #linear>
      %result_1 = ttng.tmem_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      %result_2, %token = ttng.tmem_alloc %arg7 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %8 = ttng.tc_gen5_mma_scaled %5, %7, %result_2[%token], %result, %result_1, %true, %true lhs = e4m3 rhs = e4m3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %result_3, %token_4 = ttng.tmem_load %result_2[%8] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %result_3 : tensor<128x128xf32, #blocked>
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], tt.num_stages = 2 : i64, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // SMEM-LABEL: tt.func @local_alloc_default_partition
  // SMEM: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
  // SMEM: ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // SMEM: ttg.local_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
  tt.func @local_alloc_default_partition(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x128xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    // Three cross-partition values => three semaphore pairs (6 creates)
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      // Producer for LHS TMA load

      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %lhs_trans = ttg.memdesc_trans %5 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared1, #smem> -> !ttg.memdesc<128x128xf16, #shared, #smem>

      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %lhs_trans, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SMEM-LABEL: tt.func @descriptor_store_via_convert_uses_descriptor_encoding
  // SMEM: [[DESC_CONV_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // SMEM: ttg.local_store {{.*}}, {{.*}} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // SMEM: [[DESC_CONV_LOAD:%.*]] = ttg.local_load {{.*}} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
  // SMEM: [[DESC_CONV:%.*]] = ttg.convert_layout [[DESC_CONV_LOAD]]
  // SMEM: tt.descriptor_store {{.*}}, [[DESC_CONV]]
  tt.func @descriptor_store_via_convert_uses_descriptor_encoding(%desc: !tt.tensordesc<tensor<128x128xf16, #shared>>, %lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    scf.for %i = %lb to %ub step %step : i32 {
      %val = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x128xf16, #linear>
      %cvt = ttg.convert_layout %val {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
      tt.descriptor_store %desc[%i, %c0_i32], %cvt {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  // SMEM-LABEL: tt.func @descriptor_store_direct_uses_descriptor_encoding
  // SMEM: [[DESC_DIRECT_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // SMEM: ttg.local_store {{.*}}, {{.*}} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // SMEM: [[DESC_DIRECT_LOAD:%.*]] = ttg.local_load {{.*}} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
  // SMEM: tt.descriptor_store {{.*}}, [[DESC_DIRECT_LOAD]]
  tt.func @descriptor_store_direct_uses_descriptor_encoding(%desc: !tt.tensordesc<tensor<128x128xf16, #shared>>, %lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    scf.for %i = %lb to %ub step %step : i32 {
      %val = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<128x128xf16, #blocked>
      tt.descriptor_store %desc[%i, %c0_i32], %val {ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

// SMEM-LABEL: tt.func @two_consumers
// SMEM: [[FANOUT_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
// SMEM: ttg.local_store {{.*}}, {{.*}} {ttg.partition = array<i32: 0>}
// SMEM: [[FANOUT_P1:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 1>}
// SMEM: "op_b"([[FANOUT_P1]]) {ttg.partition = array<i32: 1>}
// SMEM: [[FANOUT_P2:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 2>}
// SMEM: "op_c"([[FANOUT_P2]]) {ttg.partition = array<i32: 2>}
// SMEM: "op_d"([[FANOUT_P2]]) {ttg.partition = array<i32: 2>}
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // Producer: acquire EMPTY, buffer, store val, release FULL

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // Consumer partition 1: acquire FULL, buffer, load, release EMPTY

    "op_c"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
    "op_d"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // Producer: acquire EMPTY, buffer, store, release FULL
    %0 = "op_a"() {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ty
    // Consumer: acquire FULL, buffer, load, release EMPTY
    "op_b"(%k) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// SMEM-LABEL: tt.func @different_yield_partition
// SMEM: ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
// SMEM: ttg.local_store {{.*}} {ttg.partition = array<i32: 0>}
// SMEM: ttg.local_load {{.*}} {ttg.partition = array<i32: 1>}
// SMEM: scf.yield {{.*}} : tensor<1xi32, #blocked>
tt.func @different_yield_partition(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // Producer: acquire EMPTY, buffer, store, release FULL
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer: acquire FULL, buffer, load, release EMPTY

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// Two cross-partition iter_args => two semaphore pairs (for %k and %l)
tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
    // Producer put for %l: acquire EMPTY_L, buffer, store, release FULL_L
    // Producer put for %k: acquire EMPTY_K, buffer, store, release FULL_K

    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer for %k in partition 1
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer for %k in partition 2
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()

    // Consumer for %l in partition 1
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer for %l in partition 2
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty

  scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
    // Producer: acquire EMPTY, buffer, store %l, release FULL
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer partition 1: acquire FULL, buffer, load, release EMPTY
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [1, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// Three cross-partition iter_args => three semaphore pairs
tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // Producer puts for %c, %b, %a — all in partition 0
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer for %a in partition 1
    "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> ()

    // Consumer for %b in partition 2
    "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> ()

    // Consumer for %c in partition 3
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %a, %a : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0, 0, 0], ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @multiplicity_branch2(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // Producer puts: %c from partition 2, %b from partition 1, %a from partition 0
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer for %a in partition 1
    %d = "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> !ty

    // Consumer for %b in partition 2
    %e = "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> !ty

    // Consumer for %c in partition 3
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %d, %e : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    // Producer: acquire EMPTY, buffer, store, release FULL

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // Consumer: acquire FULL, buffer, load, release EMPTY

    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 1], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @conditional_consumer(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    // Producer: acquire EMPTY, buffer, store, release FULL
    %cond = "rand"() {ttg.partition = array<i32: 1>} : () -> i1
    // Consumer: acquire FULL, buffer, load, release EMPTY (before if)
    %1 = scf.if %cond -> !ty {
      "something"() {ttg.partition = array<i32: 1>} : () -> ()
      scf.yield {ttg.partition = array<i32: 1>} %0 : !ty
    } else {
      %2 = "something"() {ttg.partition = array<i32: 1>} : () -> !ty
      scf.yield {ttg.partition = array<i32: 1>} %2 : !ty
    } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
    "keep"(%1) {ttg.partition = array<i32: 1>} : (!ty) -> ()
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @no_def_op(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    // Producer: acquire EMPTY, buffer, splat, store, release FULL
    // Consumer: acquire FULL, buffer, load, release EMPTY, unsplat
    arith.addi %k, %k {ttg.partition = array<i32: 1>} : i32
    scf.yield {ttg.partition = array<i32: 0>} %k : i32
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
  tt.return
}

// SMEM-LABEL: tt.func @scalar_consumers
// SMEM: [[SCALAR_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
// SMEM: [[SCALAR:%.*]] = "op_a"() {ttg.partition = array<i32: 0>} : () -> i32
// SMEM: [[SPLAT:%.*]] = tt.splat [[SCALAR]] {ttg.partition = array<i32: 0>} : i32 -> tensor<1xi32, #blocked>
// SMEM: ttg.local_store [[SPLAT]], {{.*}} {ttg.partition = array<i32: 0>}
// SMEM: [[SCALAR_LOAD:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 1>}
// SMEM: [[UNSPLAT:%.*]] = tt.unsplat [[SCALAR_LOAD]] {ttg.partition = array<i32: 1>}
// SMEM: "op_b"([[UNSPLAT]]) {ttg.partition = array<i32: 1>}
tt.func @scalar_consumers(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> i32
    // Producer: acquire EMPTY, buffer, splat scalar, store, release FULL

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (i32) -> ()
    // Consumer: acquire FULL, buffer, load, release EMPTY, unsplat

  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}


}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // Two cross-partition values => two semaphore pairs

  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // Producer for %0→partition1: acquire EMPTY1, buffer, store, release FULL1

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // Consumer: acquire FULL1, buffer, load, release EMPTY1
    // Producer for %1→partition0: acquire EMPTY2, buffer, store, release FULL2

    // Consumer: acquire FULL2, buffer, load, release EMPTY2
    "op_c"(%1) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    scf.yield
  } {tt.warp_specialize, ttg.partition.stages = [0, 2], ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // Three cross-partition values => three semaphore pairs
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty

    %2 = "op_c"(%1) {ttg.partition = array<i32: 2>} : (!ty) -> !ty

    "op_c"(%2) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    scf.yield
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 3], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}


// -----

// Two cross-partition values (outer LHS + inner RHS)
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @inner_loop_fixed_operand(%arg0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %arg3, %c128_i32 : i32
    %2 = arith.divsi %arg4, %c128_i32 : i32
    %3 = arith.divsi %arg5, %c128_i32 : i32
    %4 = arith.muli %1, %2 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %result, %token = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Producer for outer LHS TMA load
    // Consumer for outer LHS
    // Producer for inner RHS TMA load
    // Consumer for inner RHS
    %6 = scf.for %arg6 = %0 to %4 step %c148_i32 iter_args(%arg7 = %token) -> (!ttg.async.token)  : i32 {
      %7 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.muli %7, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.subi %1, %8 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.minsi %9, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %11 = arith.remsi %arg6, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %12 = arith.addi %8, %11 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.divsi %13, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %15 = arith.muli %12, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = arith.muli %14, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %17 = tt.descriptor_load %arg0[%15, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %18 = ttg.local_alloc %17 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %19:2 = scf.for %arg8 = %c0_i32 to %3 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %arg7) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg8, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg1[%16, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = ttg.memdesc_trans %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %26 = ttng.tc_gen5_mma %18, %25, %result[%arg10], %arg9, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %26 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %20 = tt.fp_to_fp %result_0, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %21 = ttg.convert_layout %20 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %arg2[%15, %16], %21 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %token_1 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
// SMEM-LABEL: tt.func @semaphore_result_outside_scheduled_loop
// SMEM: [[OUTSIDE_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
// SMEM: ttg.local_store {{.*}}, {{.*}} {ttg.partition = array<i32: 2>}
// SMEM: [[OUTSIDE_LOAD:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 0>}
// SMEM: "op_b"([[OUTSIDE_LOAD]]) {ttg.partition = array<i32: 0>}
tt.func @semaphore_result_outside_scheduled_loop(%lb: i32, %ub: i32, %step: i32) {
  // Producer: acquire EMPTY, buffer, store, release FULL
  // Consumer: acquire FULL, buffer, load, release EMPTY
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 2>} : () -> !ty
    "op_b"(%0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    scf.for %j = %lb to %ub step %step : i32 {
      %x = arith.addi %lb, %lb {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      scf.yield
    } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0>}
    scf.yield
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 2>, ttg.partition.stages = [0, 1], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}
}

// -----

#offs = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SMEM-LABEL: tt.func @descriptor_gather_alloca
  // SMEM: [[GATHER_BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // SMEM: nvws.descriptor_gather {{.*}} [[GATHER_BUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // SMEM: [[GATHER_LOAD:%.*]] = ttg.local_load [[GATHER_BUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked1>
  // SMEM: "use"([[GATHER_LOAD]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
  tt.func @descriptor_gather_alloca(%desc: !tt.tensordesc<tensor<1x64xf16, #shared>>, %idx: tensor<128xi32, #offs>, %lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = tt.descriptor_gather %desc[%idx, %i] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #offs>, i32) -> tensor<128x64xf16, #blocked>
      "use"(%v) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // TMEM-LABEL: tt.func @rank1_fp_ssa_uses_tmem_alloca
  // TMEM: [[TMEM_BUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x1xf32
  // TMEM: tt.expand_dims {{.*}} {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
  // TMEM: ttng.tmem_store {{.*}}, [[TMEM_BUF]],
  // TMEM: [[TMEM_LOAD:%.*]], {{.*}} = ttng.tmem_load [[TMEM_BUF]]
  // TMEM: [[TMEM_RESHAPED:%.*]] = tt.reshape [[TMEM_LOAD]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
  // TMEM: [[TMEM_VALUE:%.*]] = ttg.convert_layout [[TMEM_RESHAPED]]
  // TMEM: "use"([[TMEM_VALUE]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
  tt.func @rank1_fp_ssa_uses_tmem_alloca(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> tensor<128xf32, #blocked>
      "use"(%v) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128xf32, #blocked>) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
