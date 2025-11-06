// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc -canonicalize | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc="hoist-out-of-if=true" -canonicalize | FileCheck %s -check-prefix=HOIST-IF

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_mma
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK-NOT: ttng.tmem_load
  // CHECK:   "end_of_loop"
  // CHECK:   yield %[[MMA_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @chained_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "end_of_loop"() : () -> ()
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[ACC:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[MMA_TOK]]]
  // CHECK:   %[[ACC_MUL:.*]] = arith.mulf %[[ACC]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store %[[ACC_MUL]], %[[ACC_TM]][%[[LOAD_TOK]]], %[[TRUE]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @changed_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc_if = arith.mulf %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @changed_acc_before_mma
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK:   %[[ACC:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[ACC_MUL:.*]] = arith.mulf %[[ACC]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store %[[ACC_MUL]], %[[ACC_TM]][%[[LOAD_TOK]]], %[[TRUE]]
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[STORE_TOK]]]
  // CHECK:   yield %[[MMA_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @changed_acc_before_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_mul = arith.mulf %acc, %cst2 : tensor<128x128xf32, #blocked>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc_mul : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[CND:.*]] = "cnd"() : () -> i1
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK-NOT: ttng.tmem_load
  // CHECK:   %[[CND_NEG:.*]] = arith.xori %[[CND]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store {{.*}}, %[[ACC_TM]][%[[MMA_TOK]]], %[[CND_NEG]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @select_after_mma(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cnd = "cnd"() : () -> i1
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc_if = arith.select %cnd, %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  // CHECK: %[[ACC_TM1:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[ACC_TM2:.*]] = ttng.tmem_alloc : ()
  // CHECK: scf.for
  // CHECK:   ttng.tmem_store
  // CHECK:   ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  // CHECK:   ttng.tmem_store
  // CHECK:   ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  tt.func public @two_dots(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %acc_ptr: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %res_ptr: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: i32) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %3 = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc = tt.load %acc_ptr : tensor<128x128x!tt.ptr<f32>, #blocked>

      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %4, %6, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      %acc_tm2, %acc_tok2 = ttng.tmem_alloc %acc_res : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok2 = ttng.tc_gen5_mma %4, %6, %acc_tm2[%acc_tok2], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res2, %load_tok2 = ttng.tmem_load %acc_tm2[%mma_tok2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      tt.store %res_ptr, %acc_res2 : tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hoist_constant_inputs
  tt.func public @hoist_constant_inputs(%arg0: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %arg1: !ttg.memdesc<64x128xi8, #shared1, #smem>, %arg2: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, %arg3: i32, %arg4: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: arith.trunci
    // CHECK: tt.splat
    // CHECK: ttng.tmem_alloc
    // CHECK: scf.for
    // CHECK:  ttng.tc_gen5_mma_scaled
    scf.for %arg5 = %c0_i32 to %arg3 step %c1_i32  : i32 {
      %0 = arith.trunci %arg3 : i32 to i8
      %1 = tt.splat %0 : i8 -> tensor<128x4xi8, #blocked1>
      %2 = ttng.tmem_alloc %1 : (tensor<128x4xi8, #blocked1>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %arg0, %arg1, %arg4, %arg2, %2, %true, %true lhs = e5m2 rhs = e2m1 : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<64x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @use_in_conditional
  // CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  // CHECK: %[[CND:.*]] = "cnd"() : () -> i1
  // CHECK: %[[ACC_TM:.*]], %[[ALLOC_TOK:.*]] = ttng.tmem_alloc : ()
  // CHECK: %[[INIT_TOK:.*]] = ttng.tmem_store %[[C0]], %[[ACC_TM]][%[[ALLOC_TOK]]]
  // CHECK: %[[RES_TOK:.*]] = scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[INIT_TOK]])
  // CHECK-NOT: ttng.tmem_alloc
  // CHECK-NOT: ttng.tmem_store
  // CHECK:   %[[MMA_TOK:.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[ACC_TM]][%[[TOK]]]
  // CHECK:   %[[CND_TOK:.*]] = scf.if %[[CND]]
  // CHECK:     "epilogue"()
  // CHECK:     %[[RESULT:.*]], %[[LOAD_TOK:.*]] = ttng.tmem_load %[[ACC_TM]][%[[MMA_TOK]]]
  // CHECK:     yield %[[LOAD_TOK]]
  // CHECK:   else
  // CHECK:     yield %[[MMA_TOK]]
  // CHECK:   %[[CND_NEG:.*]] = arith.xori %[[CND]]
  // CHECK:   %[[STORE_TOK:.*]] = ttng.tmem_store {{.*}}, %[[ACC_TM]][%[[CND_TOK]]], %[[CND_NEG]]
  // CHECK:   yield %[[STORE_TOK]]
  // CHECK: %[[ACC_TM_LOAD:.*]], %{{.*}} = ttng.tmem_load %[[ACC_TM]][%[[RES_TOK]]]
  // CHECK: arith.truncf %[[ACC_TM_LOAD]]
  tt.func public @use_in_conditional(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cnd = "cnd"() : () -> i1
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.if %cnd {
        "epilogue"() : () -> ()
        "user"(%acc_res) : (tensor<128x128xf32, #blocked>) -> ()
      }
      %acc_if = arith.select %cnd, %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %acc_if : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // HOIST-IF-LABEL: @hoist_out_of_if
  tt.func public @hoist_out_of_if(%arg0: i1, %arg1: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    // HOIST-IF: %[[A:.+]], %[[T0:.+]] = ttng.tmem_alloc : ()
    // HOIST-IF: %[[T1:.+]] = ttng.tmem_store %{{.*}}, %[[A]][%[[T0]]]
    // HOIST-IF: %[[I:.+]] = scf.if %{{.+}} -> (!ttg.async.token) {
    // HOIST-IF:   %[[T2:.+]] = "write_to_tmem"
    // HOIST-IF:   scf.yield %[[T2]]
    // HOIST-IF: } else {
    // HOIST-IF:   scf.yield %[[T1]]
    // HOIST-IF: }
    // HOIST-IF: %[[L:.+]], %[[T4:.+]] = ttng.tmem_load %[[A]][%[[I]]
    // HOIST-IF: tt.return %[[L]]
    %0 = scf.if %arg0 -> (tensor<128x128xf32, #blocked>) {
      %result, %token = ttng.tmem_alloc %arg1 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %1 = "write_to_tmem"(%result) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> !ttg.async.token
      %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %result_0 : tensor<128x128xf32, #blocked>
    } else {
      scf.yield %arg1 : tensor<128x128xf32, #blocked>
    }
    tt.return %0 : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @forward_tmem_load(%m: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %t: !ttg.async.token) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) {
    %true = arith.constant true
    %result, %token0 = ttng.tmem_load %m[%t] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // HOIST-IF-LABEL: @forward_tmem_load
    // HOIST-IF-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32,
    // HOIST-IF-SAME:    %[[ARG1:.+]]: !ttg.async.token
    // HOIST-IF-NEXT:    tt.return %[[ARG0]], %[[ARG1]]
    %result1, %token1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %token2 = ttng.tmem_store %result, %result1[%token1], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return %result1, %token2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sink_multiple_tmem_load
  tt.func public @sink_multiple_tmem_load(%m: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %t: !ttg.async.token) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %res:2 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%init0 = %cst, %init1 = %cst) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>)  : i32 {
      // Any order is fine, just make sure we don't reorder them in an infinite loop.
      // CHECK-COUNT-2: ttng.tmem_load
      // CHECK: scf.yield
      %l0, %token_1 = ttng.tmem_load %m[%t] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %l1, %token_2 = ttng.tmem_load %m[%t] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %l0, %l1 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 3 : i32}
    tt.return %res#0, %res#1 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @_attn_fwd_persist
    tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64, %N_CTX: i32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_17 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_18 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %n_tile_num_19 = arith.addi %N_CTX, %n_tile_num : i32
    %n_tile_num_20 = arith.divsi %n_tile_num_19, %c256_i32 : i32
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %n_tile_num_20, %Z : i32
    %total_tiles_21 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_21, %num_progs : i32
    %0 = arith.remsi %total_tiles_21, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_22 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_22 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %offset_y = arith.muli %N_CTX, %H : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m1 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked>
    // CHECK: scf.for
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_22 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_22, %n_tile_num_20 : i32
      %off_hz = arith.divsi %tile_idx_22, %n_tile_num_20 : i32
      %off_z = arith.divsi %off_hz, %H : i32
      %off_h = arith.remsi %off_hz, %H : i32
      %offset_y_23 = arith.muli %off_z, %offset_y : i32
      %offset_y_24 = arith.muli %off_h, %N_CTX : i32
      %offset_y_25 = arith.addi %offset_y_23, %offset_y_24 : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 : i32
      %qo_offset_y_26 = arith.addi %offset_y_25, %qo_offset_y : i32
      %offs_m0_27 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_28 = arith.addi %offs_m0_27, %offs_m0 : tensor<128xi32, #blocked1>
      %offs_m1_29 = arith.addi %offs_m0_27, %offs_m1 : tensor<128xi32, #blocked1>
      %q0 = tt.descriptor_load %desc_q[%qo_offset_y_26, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q0_30 = ttg.local_alloc %q0 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %q1 = arith.addi %qo_offset_y_26, %c128_i32 : i32
      %q1_31 = tt.descriptor_load %desc_q[%q1, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q1_32 = ttg.local_alloc %q1_31 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      // CHECK-COUNT-4: ttng.tmem_alloc
      // CHECK-NOT: ttng.tmem_store
      // CHECK: scf.for
      // CHECK-NOT: ttng.tmem_alloc
      // CHECK-COUNT-2: ttng.tmem_load
      // CHECK: ttng.tmem_store
      // CHECK-COUNT-2: ttng.tmem_load
      // CHECK: ttng.tmem_store
      // CHECK-COUNT-2: ttng.tmem_load
      %offsetkv_y:9 = scf.for %offsetkv_y_43 = %c0_i32 to %N_CTX step %c128_i32 iter_args(%arg28 = %cst_16, %arg29 = %cst_16, %arg30 = %cst_18, %arg31 = %cst_18, %arg32 = %cst_17, %arg33 = %cst_17, %offset_y_44 = %offset_y_25, %acc = %false, %acc_45 = %false) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, i1)  : i32 {
        %k = tt.descriptor_load %desc_k[%offset_y_44, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %k_46 = ttg.local_alloc %k : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %k_47 = ttg.memdesc_trans %k_46 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %v = tt.descriptor_load %desc_v[%offset_y_44, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %v_48 = ttg.local_alloc %v : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %qk_49, %qk_50 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

        %qk_51 = ttng.tc_gen5_mma %q0_30, %k_47, %qk_49[%qk_50], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_52, %qk_53 = ttng.tmem_load %qk_49[%qk_51] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_54 = "tt.reduce"(%qk_52) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_114: f32, %m_ij_115: f32):
          %m_ij_116 = arith.maxnumf %m_ij_114, %m_ij_115 : f32
          tt.reduce.return %m_ij_116 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_55 = arith.mulf %m_ij_54, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_56 = arith.maxnumf %arg32, %m_ij_55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_57 = arith.mulf %qk_52, %qk : tensor<128x128xf32, #blocked>
        %qk_58 = tt.expand_dims %m_ij_56 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_59 = tt.broadcast %qk_58 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_60 = arith.subf %qk_57, %qk_59 : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_60 : tensor<128x128xf32, #blocked>
        %alpha = arith.subf %arg32, %m_ij_56 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_61 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_114: f32, %l_ij_115: f32):
          %l_ij_116 = arith.addf %l_ij_114, %l_ij_115 : f32
          tt.reduce.return %l_ij_116 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %9 = tt.reshape %arg28 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
        %10 = tt.trans %9 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
        %outLHS, %outRHS = tt.split %10 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
        %acc0_62 = tt.expand_dims %alpha_61 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_63 = ttg.convert_layout %acc0_62 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5>
        %acc0_64 = tt.broadcast %acc0_63 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5>
        %acc0_65 = arith.mulf %outLHS, %acc0_64 : tensor<128x64xf32, #blocked5>
        %acc1_66 = arith.mulf %outRHS, %acc0_64 : tensor<128x64xf32, #blocked5>
        %acc_67 = tt.join %acc0_65, %acc1_66 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4>
        %acc_68 = tt.trans %acc_67 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3>
        %acc_69 = tt.reshape %acc_68 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked>
        %p_70 = arith.truncf %p : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %p_71 = ttg.local_alloc %p_70 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %acc_72, %acc_73 = ttng.tmem_alloc %acc_69 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_74 = ttng.tc_gen5_mma %p_71, %v_48, %acc_72[%acc_73], %acc, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_75, %acc_76 = ttng.tmem_load %acc_72[%acc_74] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %l_i = arith.mulf %arg30, %alpha_61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_77 = arith.addf %l_i, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_78, %qk_79 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %qk_80 = ttng.tc_gen5_mma %q1_32, %k_47, %qk_78[%qk_79], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
                %qk_81, %qk_82 = ttng.tmem_load %qk_78[%qk_80] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_83 = "tt.reduce"(%qk_81) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_114: f32, %m_ij_115: f32):
          %m_ij_116 = arith.maxnumf %m_ij_114, %m_ij_115 : f32
          tt.reduce.return %m_ij_116 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_84 = arith.mulf %m_ij_83, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_85 = arith.maxnumf %arg33, %m_ij_84 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_86 = arith.mulf %qk_81, %qk : tensor<128x128xf32, #blocked>
        %qk_87 = tt.expand_dims %m_ij_85 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_88 = tt.broadcast %qk_87 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_89 = arith.subf %qk_86, %qk_88 : tensor<128x128xf32, #blocked>
        %p_90 = math.exp2 %qk_89 : tensor<128x128xf32, #blocked>
        %alpha_91 = arith.subf %arg33, %m_ij_85 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_92 = math.exp2 %alpha_91 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_93 = "tt.reduce"(%p_90) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_114: f32, %l_ij_115: f32):
          %l_ij_116 = arith.addf %l_ij_114, %l_ij_115 : f32
          tt.reduce.return %l_ij_116 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %11 = tt.reshape %arg29 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
        %12 = tt.trans %11 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
        %outLHS_94, %outRHS_95 = tt.split %12 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
        %acc0_96 = tt.expand_dims %alpha_92 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_97 = ttg.convert_layout %acc0_96 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5>
        %acc0_98 = tt.broadcast %acc0_97 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5>
        %acc0_99 = arith.mulf %outLHS_94, %acc0_98 : tensor<128x64xf32, #blocked5>
        %acc1_100 = arith.mulf %outRHS_95, %acc0_98 : tensor<128x64xf32, #blocked5>
        %acc_101 = tt.join %acc0_99, %acc1_100 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4>
        %acc_102 = tt.trans %acc_101 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3>
        %acc_103 = tt.reshape %acc_102 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked>
        %p_104 = arith.truncf %p_90 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %p_105 = ttg.local_alloc %p_104 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %acc_106, %acc_107 = ttng.tmem_alloc %acc_103 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_108 = ttng.tc_gen5_mma %p_105, %v_48, %acc_106[%acc_107], %acc_45, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_109, %acc_110 = ttng.tmem_load %acc_106[%acc_108] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %l_i_111 = arith.mulf %arg31, %alpha_92 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_112 = arith.addf %l_i_111, %l_ij_93 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_113 = arith.addi %offset_y_44, %c128_i32 : i32
        scf.yield %acc_75, %acc_109, %l_i_77, %l_i_112, %m_ij_56, %m_ij_85, %offsetkv_y_113, %true, %true : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, i1
      } {tt.disallow_acc_multi_buffer}
      %m_i0 = math.log2 %offsetkv_y#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i0_33 = arith.addf %offsetkv_y#4, %m_i0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc0 = tt.expand_dims %offsetkv_y#2 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_34 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc0_35 = arith.divf %offsetkv_y#0, %acc0_34 : tensor<128x128xf32, #blocked>
      %m_ptrs0 = arith.muli %off_hz, %N_CTX : i32
      %m_ptrs0_36 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32
      %m_ptrs0_37 = tt.splat %m_ptrs0_36 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
      %m_ptrs0_38 = tt.addptr %m_ptrs0_37, %offs_m0_28 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      %3 = ttg.convert_layout %m_i0_33 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      tt.store %m_ptrs0_38, %3 : tensor<128x!tt.ptr<f32>, #blocked1>
      %4 = arith.truncf %acc0_35 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %5 = ttg.convert_layout %4 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
      tt.descriptor_store %desc_o[%qo_offset_y_26, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
      %m_i1 = math.log2 %offsetkv_y#3 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_i1_39 = arith.addf %offsetkv_y#5, %m_i1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc1 = tt.expand_dims %offsetkv_y#3 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc1_40 = tt.broadcast %acc1 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %acc1_41 = arith.divf %offsetkv_y#1, %acc1_40 : tensor<128x128xf32, #blocked>
      %m_ptrs1 = tt.addptr %m_ptrs0_37, %offs_m1_29 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
      %6 = ttg.convert_layout %m_i1_39 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
      tt.store %m_ptrs1, %6 : tensor<128x!tt.ptr<f32>, #blocked1>
      %7 = arith.truncf %acc1_41 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %8 = ttg.convert_layout %7 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
      tt.descriptor_store %desc_o[%q1, %c0_i32], %8 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
      %tile_idx_42 = arith.addi %tile_idx_22, %num_progs : i32
      scf.yield %tile_idx_42 : i32
    } {tt.warp_specialize}
    tt.return
  }
}
