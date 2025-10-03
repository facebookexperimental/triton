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
#loc = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":415:0)
#loc1 = loc(unknown)
#loc15 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":468:12)
#loc20 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":144:12)
#loc21 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":316:12)
#loc44 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":53:42)
#loc53 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":58:21)
#loc68 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":159:12)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#loc92 = loc("sm_scale"(#loc))
#loc93 = loc("M"(#loc))
#loc94 = loc("Z"(#loc))
#loc95 = loc("H"(#loc))
#loc96 = loc("desc_q"(#loc))
#loc97 = loc("desc_k"(#loc))
#loc98 = loc("desc_v"(#loc))
#loc99 = loc("desc_o"(#loc))
#loc100 = loc("N_CTX"(#loc))
#loc113 = loc(callsite(#loc21 at #loc15))
#loc135 = loc("m_ij"(#loc44))
#loc142 = loc("l_ij"(#loc53))
#loc181 = loc(callsite(#loc20 at #loc113))
#loc198 = loc(callsite(#loc68 at #loc113))
#loc217 = loc(callsite(#loc135 at #loc181))
#loc224 = loc(callsite(#loc142 at #loc181))
#loc239 = loc(callsite(#loc135 at #loc198))
#loc248 = loc(callsite(#loc142 at #loc198))
#loc264 = loc(callsite(#loc1 at #loc217))
#loc266 = loc(callsite(#loc1 at #loc224))
#loc268 = loc(callsite(#loc1 at #loc239))
#loc270 = loc(callsite(#loc1 at #loc248))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32 loc("sm_scale"(#loc)), %M: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("M"(#loc)), %Z: i32 loc("Z"(#loc)), %H: i32 {tt.divisibility = 16 : i32} loc("H"(#loc)), %desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>> loc("desc_q"(#loc)), %desc_q_0: i32 loc("desc_q"(#loc)), %desc_q_1: i32 loc("desc_q"(#loc)), %desc_q_2: i64 loc("desc_q"(#loc)), %desc_q_3: i64 loc("desc_q"(#loc)), %desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>> loc("desc_k"(#loc)), %desc_k_4: i32 loc("desc_k"(#loc)), %desc_k_5: i32 loc("desc_k"(#loc)), %desc_k_6: i64 loc("desc_k"(#loc)), %desc_k_7: i64 loc("desc_k"(#loc)), %desc_v: !tt.tensordesc<tensor<128x128xbf16, #shared>> loc("desc_v"(#loc)), %desc_v_8: i32 loc("desc_v"(#loc)), %desc_v_9: i32 loc("desc_v"(#loc)), %desc_v_10: i64 loc("desc_v"(#loc)), %desc_v_11: i64 loc("desc_v"(#loc)), %desc_o: !tt.tensordesc<tensor<128x128xbf16, #shared>> loc("desc_o"(#loc)), %desc_o_12: i32 loc("desc_o"(#loc)), %desc_o_13: i32 loc("desc_o"(#loc)), %desc_o_14: i64 loc("desc_o"(#loc)), %desc_o_15: i64 loc("desc_o"(#loc)), %N_CTX: i32 loc("N_CTX"(#loc))) attributes {noinline = false} {
    %false = arith.constant false loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %n_tile_num = arith.constant 255 : i32 loc(#loc173)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst = arith.constant 1.44269502 : f32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked> loc(#loc1)
    %cst_17 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc1)
    %cst_18 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc1)
    %n_tile_num_19 = arith.addi %N_CTX, %n_tile_num : i32 loc(#loc173)
    %n_tile_num_20 = arith.divsi %n_tile_num_19, %c256_i32 : i32 loc(#loc174)
    %prog_id = tt.get_program_id x : i32 loc(#loc102)
    %num_progs = tt.get_num_programs x : i32 loc(#loc103)
    %total_tiles = arith.muli %n_tile_num_20, %Z : i32 loc(#loc104)
    %total_tiles_21 = arith.muli %total_tiles, %H : i32 loc(#loc105)
    %tiles_per_sm = arith.divsi %total_tiles_21, %num_progs : i32 loc(#loc175)
    %0 = arith.remsi %total_tiles_21, %num_progs : i32 loc(#loc10)
    %1 = arith.cmpi slt, %prog_id, %0 : i32 loc(#loc11)
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_22 = arith.addi %tiles_per_sm, %c1_i32 : i32 loc(#loc176)
      scf.yield %tiles_per_sm_22 : i32 loc(#loc176)
    } else {
      scf.yield %tiles_per_sm : i32 loc(#loc1)
    } loc(#loc12)
    %offset_y = arith.muli %N_CTX, %H : i32 loc(#loc177)
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1> loc(#loc178)
    %offs_m1 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1> loc(#loc179)
    %qk_scale = arith.mulf %sm_scale, %cst : f32 loc(#loc180)
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc213)
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked> loc(#loc214)
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_22 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_22, %n_tile_num_20 : i32 loc(#loc116)
      %off_hz = arith.divsi %tile_idx_22, %n_tile_num_20 : i32 loc(#loc117)
      %off_z = arith.divsi %off_hz, %H : i32 loc(#loc182)
      %off_h = arith.remsi %off_hz, %H : i32 loc(#loc183)
      %offset_y_23 = arith.muli %off_z, %offset_y : i32 loc(#loc184)
      %offset_y_24 = arith.muli %off_h, %N_CTX : i32 loc(#loc185)
      %offset_y_25 = arith.addi %offset_y_23, %offset_y_24 : i32 loc(#loc186)
      %qo_offset_y = arith.muli %pid, %c256_i32 : i32 loc(#loc187)
      %qo_offset_y_26 = arith.addi %offset_y_25, %qo_offset_y : i32 loc(#loc188)
      %offs_m0_27 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked1> loc(#loc189)
      %offs_m0_28 = arith.addi %offs_m0_27, %offs_m0 : tensor<128xi32, #blocked1> loc(#loc189)
      %offs_m1_29 = arith.addi %offs_m0_27, %offs_m1 : tensor<128xi32, #blocked1> loc(#loc190)
      %q0 = tt.descriptor_load %desc_q[%qo_offset_y_26, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2> loc(#loc191)
      %q0_30 = ttg.local_alloc %q0 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc191)
      %q1 = arith.addi %qo_offset_y_26, %c128_i32 : i32 loc(#loc192)
      %q1_31 = tt.descriptor_load %desc_q[%q1, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2> loc(#loc193)
      %q1_32 = ttg.local_alloc %q1_31 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc193)
      %offsetkv_y:9 = scf.for %offsetkv_y_43 = %c0_i32 to %N_CTX step %c128_i32 iter_args(%arg28 = %cst_16, %arg29 = %cst_16, %arg30 = %cst_18, %arg31 = %cst_18, %arg32 = %cst_17, %arg33 = %cst_17, %offset_y_44 = %offset_y_25, %acc = %false, %acc_45 = %false) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, i1)  : i32 {
        %k = tt.descriptor_load %desc_k[%offset_y_44, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2> loc(#loc195)
        %k_46 = ttg.local_alloc %k : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc196)
        %k_47 = ttg.memdesc_trans %k_46 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem> loc(#loc196)
        %v = tt.descriptor_load %desc_v[%offset_y_44, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2> loc(#loc197)
        %v_48 = ttg.local_alloc %v : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc197)
        %qk_49, %qk_50 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc216)
        %qk_51 = ttng.tc_gen5_mma %q0_30, %k_47, %qk_49[%qk_50], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc216)
        %qk_52, %qk_53 = ttng.tmem_load %qk_49[%qk_51] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc216)
        %m_ij_54 = "tt.reduce"(%qk_52) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_114: f32 loc(callsite(#loc1 at #loc217)), %m_ij_115: f32 loc(callsite(#loc1 at #loc217))):
          %m_ij_116 = arith.maxnumf %m_ij_114, %m_ij_115 : f32 loc(#loc272)
          tt.reduce.return %m_ij_116 : f32 loc(#loc263)
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc263)
        %m_ij_55 = arith.mulf %m_ij_54, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc213)
        %m_ij_56 = arith.maxnumf %arg32, %m_ij_55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc218)
        %qk_57 = arith.mulf %qk_52, %qk : tensor<128x128xf32, #blocked> loc(#loc214)
        %qk_58 = tt.expand_dims %m_ij_56 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc219)
        %qk_59 = tt.broadcast %qk_58 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked> loc(#loc220)
        %qk_60 = arith.subf %qk_57, %qk_59 : tensor<128x128xf32, #blocked> loc(#loc220)
        %p = math.exp2 %qk_60 : tensor<128x128xf32, #blocked> loc(#loc221)
        %alpha = arith.subf %arg32, %m_ij_56 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc222)
        %alpha_61 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc223)
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_114: f32 loc(callsite(#loc1 at #loc224)), %l_ij_115: f32 loc(callsite(#loc1 at #loc224))):
          %l_ij_116 = arith.addf %l_ij_114, %l_ij_115 : f32 loc(#loc273)
          tt.reduce.return %l_ij_116 : f32 loc(#loc265)
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc265)
        %9 = tt.reshape %arg28 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3> loc(#loc225)
        %10 = tt.trans %9 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4> loc(#loc226)
        %outLHS, %outRHS = tt.split %10 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5> loc(#loc227)
        %acc0_62 = tt.expand_dims %alpha_61 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc228)
        %acc0_63 = ttg.convert_layout %acc0_62 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5> loc(#loc229)
        %acc0_64 = tt.broadcast %acc0_63 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5> loc(#loc229)
        %acc0_65 = arith.mulf %outLHS, %acc0_64 : tensor<128x64xf32, #blocked5> loc(#loc229)
        %acc1_66 = arith.mulf %outRHS, %acc0_64 : tensor<128x64xf32, #blocked5> loc(#loc230)
        %acc_67 = tt.join %acc0_65, %acc1_66 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4> loc(#loc231)
        %acc_68 = tt.trans %acc_67 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3> loc(#loc232)
        %acc_69 = tt.reshape %acc_68 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked> loc(#loc233)
        %p_70 = arith.truncf %p : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked> loc(#loc234)
        %p_71 = ttg.local_alloc %p_70 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc234)
        %acc_72, %acc_73 = ttng.tmem_alloc %acc_69 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc235)
        %acc_74 = ttng.tc_gen5_mma %p_71, %v_48, %acc_72[%acc_73], %acc, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc235)
        %acc_75, %acc_76 = ttng.tmem_load %acc_72[%acc_74] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc235)
        %l_i = arith.mulf %arg30, %alpha_61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc236)
        %l_i_77 = arith.addf %l_i, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc237)
        %qk_78, %qk_79 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc238)
        %qk_80 = ttng.tc_gen5_mma %q1_32, %k_47, %qk_78[%qk_79], %false, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc238)
        %qk_81, %qk_82 = ttng.tmem_load %qk_78[%qk_80] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc238)
        %m_ij_83 = "tt.reduce"(%qk_81) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_114: f32 loc(callsite(#loc1 at #loc239)), %m_ij_115: f32 loc(callsite(#loc1 at #loc239))):
          %m_ij_116 = arith.maxnumf %m_ij_114, %m_ij_115 : f32 loc(#loc274)
          tt.reduce.return %m_ij_116 : f32 loc(#loc267)
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc267)
        %m_ij_84 = arith.mulf %m_ij_83, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc240)
        %m_ij_85 = arith.maxnumf %arg33, %m_ij_84 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc241)
        %qk_86 = arith.mulf %qk_81, %qk : tensor<128x128xf32, #blocked> loc(#loc242)
        %qk_87 = tt.expand_dims %m_ij_85 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc243)
        %qk_88 = tt.broadcast %qk_87 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked> loc(#loc244)
        %qk_89 = arith.subf %qk_86, %qk_88 : tensor<128x128xf32, #blocked> loc(#loc244)
        %p_90 = math.exp2 %qk_89 : tensor<128x128xf32, #blocked> loc(#loc245)
        %alpha_91 = arith.subf %arg33, %m_ij_85 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc246)
        %alpha_92 = math.exp2 %alpha_91 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc247)
        %l_ij_93 = "tt.reduce"(%p_90) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_114: f32 loc(callsite(#loc1 at #loc248)), %l_ij_115: f32 loc(callsite(#loc1 at #loc248))):
          %l_ij_116 = arith.addf %l_ij_114, %l_ij_115 : f32 loc(#loc275)
          tt.reduce.return %l_ij_116 : f32 loc(#loc269)
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc269)
        %11 = tt.reshape %arg29 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3> loc(#loc249)
        %12 = tt.trans %11 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4> loc(#loc250)
        %outLHS_94, %outRHS_95 = tt.split %12 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5> loc(#loc251)
        %acc0_96 = tt.expand_dims %alpha_92 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc252)
        %acc0_97 = ttg.convert_layout %acc0_96 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5> loc(#loc253)
        %acc0_98 = tt.broadcast %acc0_97 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5> loc(#loc253)
        %acc0_99 = arith.mulf %outLHS_94, %acc0_98 : tensor<128x64xf32, #blocked5> loc(#loc253)
        %acc1_100 = arith.mulf %outRHS_95, %acc0_98 : tensor<128x64xf32, #blocked5> loc(#loc254)
        %acc_101 = tt.join %acc0_99, %acc1_100 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4> loc(#loc255)
        %acc_102 = tt.trans %acc_101 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3> loc(#loc256)
        %acc_103 = tt.reshape %acc_102 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked> loc(#loc257)
        %p_104 = arith.truncf %p_90 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked> loc(#loc258)
        %p_105 = ttg.local_alloc %p_104 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem> loc(#loc258)
        %acc_106, %acc_107 = ttng.tmem_alloc %acc_103 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc259)
        %acc_108 = ttng.tc_gen5_mma %p_105, %v_48, %acc_106[%acc_107], %acc_45, %true : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc259)
        %acc_109, %acc_110 = ttng.tmem_load %acc_106[%acc_108] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc259)
        %l_i_111 = arith.mulf %arg31, %alpha_92 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc260)
        %l_i_112 = arith.addf %l_i_111, %l_ij_93 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc261)
        %offsetkv_y_113 = arith.addi %offset_y_44, %c128_i32 : i32 loc(#loc199)
        scf.yield %acc_75, %acc_109, %l_i_77, %l_i_112, %m_ij_56, %m_ij_85, %offsetkv_y_113, %true, %true : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, i1 loc(#loc200)
      } {tt.disallow_acc_multi_buffer} loc(#loc278)
      %m_i0 = math.log2 %offsetkv_y#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc201)
      %m_i0_33 = arith.addf %offsetkv_y#4, %m_i0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc202)
      %acc0 = tt.expand_dims %offsetkv_y#2 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc203)
      %acc0_34 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked> loc(#loc204)
      %acc0_35 = arith.divf %offsetkv_y#0, %acc0_34 : tensor<128x128xf32, #blocked> loc(#loc204)
      %m_ptrs0 = arith.muli %off_hz, %N_CTX : i32 loc(#loc205)
      %m_ptrs0_36 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32 loc(#loc206)
      %m_ptrs0_37 = tt.splat %m_ptrs0_36 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1> loc(#loc207)
      %m_ptrs0_38 = tt.addptr %m_ptrs0_37, %offs_m0_28 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1> loc(#loc207)
      %3 = ttg.convert_layout %m_i0_33 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1> loc(#loc161)
      tt.store %m_ptrs0_38, %3 : tensor<128x!tt.ptr<f32>, #blocked1> loc(#loc161)
      %4 = arith.truncf %acc0_35 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked> loc(#loc162)
      %5 = ttg.convert_layout %4 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2> loc(#loc162)
      tt.descriptor_store %desc_o[%qo_offset_y_26, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2> loc(#loc163)
      %m_i1 = math.log2 %offsetkv_y#3 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc208)
      %m_i1_39 = arith.addf %offsetkv_y#5, %m_i1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc209)
      %acc1 = tt.expand_dims %offsetkv_y#3 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked> loc(#loc210)
      %acc1_40 = tt.broadcast %acc1 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked> loc(#loc211)
      %acc1_41 = arith.divf %offsetkv_y#1, %acc1_40 : tensor<128x128xf32, #blocked> loc(#loc211)
      %m_ptrs1 = tt.addptr %m_ptrs0_37, %offs_m1_29 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1> loc(#loc212)
      %6 = ttg.convert_layout %m_i1_39 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1> loc(#loc169)
      tt.store %m_ptrs1, %6 : tensor<128x!tt.ptr<f32>, #blocked1> loc(#loc169)
      %7 = arith.truncf %acc1_41 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked> loc(#loc170)
      %8 = ttg.convert_layout %7 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2> loc(#loc170)
      tt.descriptor_store %desc_o[%q1, %c0_i32], %8 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2> loc(#loc171)
      %tile_idx_42 = arith.addi %tile_idx_22, %num_progs : i32 loc(#loc172)
      scf.yield %tile_idx_42 : i32 loc(#loc90)
    } {tt.warp_specialize} loc(#loc115)
    tt.return loc(#loc91)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":41:22)
#loc3 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":435:32)
#loc4 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":41:28)
#loc5 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":436:28)
#loc6 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":437:32)
#loc7 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":438:31)
#loc8 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":438:35)
#loc9 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":440:34)
#loc10 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":441:31)
#loc11 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":441:17)
#loc12 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":441:7)
#loc13 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":442:24)
#loc14 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":270:32)
#loc16 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":273:47)
#loc17 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":274:58)
#loc18 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":286:16)
#loc19 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":53:47)
#loc22 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":54:18)
#loc23 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":446:39)
#loc24 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":447:25)
#loc25 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":448:29)
#loc26 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":267:22)
#loc27 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":268:21)
#loc28 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":270:24)
#loc29 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":270:45)
#loc30 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":270:37)
#loc31 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":271:39)
#loc32 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":271:29)
#loc33 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":273:34)
#loc34 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":274:34)
#loc35 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":288:21)
#loc36 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":289:36)
#loc37 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":289:21)
#loc38 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":124:58)
#loc39 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":128:24)
#loc40 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":128:12)
#loc41 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":129:24)
#loc42 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":46:19)
#loc43 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":189:40)
#loc45 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":168:27)
#loc46 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":53:31)
#loc47 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":54:34)
#loc48 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":54:29)
#loc49 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":55:21)
#loc50 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":57:31)
#loc51 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":57:25)
#loc52 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":291:36)
#loc54 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":261:15)
#loc55 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":65:33)
#loc56 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":65:65)
#loc57 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":65:21)
#loc58 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":66:28)
#loc59 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":66:22)
#loc60 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":67:22)
#loc61 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":68:28)
#loc62 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":68:48)
#loc63 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":68:59)
#loc64 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":73:13)
#loc65 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":75:23)
#loc66 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":78:16)
#loc67 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":78:24)
#loc69 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":162:22)
#loc70 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":162:8)
#loc71 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":346:25)
#loc72 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":346:12)
#loc73 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":347:23)
#loc74 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":347:18)
#loc75 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":348:27)
#loc76 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":348:18)
#loc77 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":348:35)
#loc78 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":349:22)
#loc79 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":350:43)
#loc80 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":350:35)
#loc81 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":352:25)
#loc82 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":352:12)
#loc83 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":353:23)
#loc84 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":353:18)
#loc85 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":354:35)
#loc86 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":355:22)
#loc87 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":356:58)
#loc88 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":356:50)
#loc89 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":470:20)
#loc90 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":470:8)
#loc91 = loc("/home/njriasan/tlx/tritonbench/tritonbench/kernels/blackwell_triton_fused_attention_dp.py":446:4)
#loc101 = loc("n_tile_num"(#loc3))
#loc102 = loc("prog_id"(#loc5))
#loc103 = loc("num_progs"(#loc6))
#loc104 = loc("total_tiles"(#loc7))
#loc105 = loc("total_tiles"(#loc8))
#loc106 = loc("tiles_per_sm"(#loc9))
#loc107 = loc("tiles_per_sm"(#loc13))
#loc108 = loc("offset_y"(#loc14))
#loc109 = loc("offs_m0"(#loc16))
#loc110 = loc("offs_m1"(#loc17))
#loc111 = loc("qk_scale"(#loc18))
#loc112 = loc("m_ij"(#loc19))
#loc114 = loc("qk"(#loc22))
#loc115 = loc("tile_idx"(#loc23))
#loc116 = loc("pid"(#loc24))
#loc117 = loc("off_hz"(#loc25))
#loc118 = loc("off_z"(#loc26))
#loc119 = loc("off_h"(#loc27))
#loc120 = loc("offset_y"(#loc28))
#loc121 = loc("offset_y"(#loc29))
#loc122 = loc("offset_y"(#loc30))
#loc123 = loc("qo_offset_y"(#loc31))
#loc124 = loc("qo_offset_y"(#loc32))
#loc125 = loc("offs_m0"(#loc33))
#loc126 = loc("offs_m1"(#loc34))
#loc127 = loc("q0"(#loc35))
#loc128 = loc("q1"(#loc36))
#loc129 = loc("q1"(#loc37))
#loc130 = loc("acc0"(#loc38))
#loc131 = loc("k"(#loc39))
#loc132 = loc("k"(#loc40))
#loc133 = loc("v"(#loc41))
#loc134 = loc("qk"(#loc42))
#loc136 = loc("m_ij"(#loc46))
#loc137 = loc("qk"(#loc47))
#loc138 = loc("qk"(#loc48))
#loc139 = loc("p"(#loc49))
#loc140 = loc("alpha"(#loc50))
#loc141 = loc("alpha"(#loc51))
#loc143 = loc("acc0"(#loc58))
#loc144 = loc("acc0"(#loc59))
#loc145 = loc("acc1"(#loc60))
#loc146 = loc("acc"(#loc61))
#loc147 = loc("acc"(#loc62))
#loc148 = loc("acc"(#loc63))
#loc149 = loc("p"(#loc64))
#loc150 = loc("acc"(#loc65))
#loc151 = loc("l_i"(#loc66))
#loc152 = loc("l_i"(#loc67))
#loc153 = loc("offsetkv_y"(#loc69))
#loc154 = loc("m_i0"(#loc71))
#loc155 = loc("m_i0"(#loc72))
#loc156 = loc("acc0"(#loc73))
#loc157 = loc("acc0"(#loc74))
#loc158 = loc("m_ptrs0"(#loc75))
#loc159 = loc("m_ptrs0"(#loc76))
#loc160 = loc("m_ptrs0"(#loc77))
#loc161 = loc(callsite(#loc78 at #loc15))
#loc162 = loc(callsite(#loc79 at #loc15))
#loc163 = loc(callsite(#loc80 at #loc15))
#loc164 = loc("m_i1"(#loc81))
#loc165 = loc("m_i1"(#loc82))
#loc166 = loc("acc1"(#loc83))
#loc167 = loc("acc1"(#loc84))
#loc168 = loc("m_ptrs1"(#loc85))
#loc169 = loc(callsite(#loc86 at #loc15))
#loc170 = loc(callsite(#loc87 at #loc15))
#loc171 = loc(callsite(#loc88 at #loc15))
#loc172 = loc("tile_idx"(#loc89))
#loc173 = loc(callsite(#loc2 at #loc101))
#loc174 = loc(callsite(#loc4 at #loc101))
#loc175 = loc("tiles_per_sm"(#loc106))
#loc176 = loc("tiles_per_sm"(#loc107))
#loc177 = loc(callsite(#loc108 at #loc15))
#loc178 = loc(callsite(#loc109 at #loc15))
#loc179 = loc(callsite(#loc110 at #loc15))
#loc180 = loc(callsite(#loc111 at #loc15))
#loc182 = loc(callsite(#loc118 at #loc15))
#loc183 = loc(callsite(#loc119 at #loc15))
#loc184 = loc(callsite(#loc120 at #loc15))
#loc185 = loc(callsite(#loc121 at #loc15))
#loc186 = loc(callsite(#loc122 at #loc15))
#loc187 = loc(callsite(#loc123 at #loc15))
#loc188 = loc(callsite(#loc124 at #loc15))
#loc189 = loc(callsite(#loc125 at #loc15))
#loc190 = loc(callsite(#loc126 at #loc15))
#loc191 = loc(callsite(#loc127 at #loc15))
#loc192 = loc(callsite(#loc128 at #loc15))
#loc193 = loc(callsite(#loc129 at #loc15))
#loc194 = loc("acc1"(#loc130))
#loc195 = loc(callsite(#loc131 at #loc113))
#loc196 = loc(callsite(#loc132 at #loc113))
#loc197 = loc(callsite(#loc133 at #loc113))
#loc199 = loc(callsite(#loc153 at #loc113))
#loc200 = loc(callsite(#loc70 at #loc113))
#loc201 = loc(callsite(#loc154 at #loc15))
#loc202 = loc(callsite(#loc155 at #loc15))
#loc203 = loc(callsite(#loc156 at #loc15))
#loc204 = loc(callsite(#loc157 at #loc15))
#loc205 = loc(callsite(#loc158 at #loc15))
#loc206 = loc(callsite(#loc159 at #loc15))
#loc207 = loc(callsite(#loc160 at #loc15))
#loc208 = loc(callsite(#loc164 at #loc15))
#loc209 = loc(callsite(#loc165 at #loc15))
#loc210 = loc(callsite(#loc166 at #loc15))
#loc211 = loc(callsite(#loc167 at #loc15))
#loc212 = loc(callsite(#loc168 at #loc15))
#loc213 = loc(callsite(#loc112 at #loc181))
#loc214 = loc(callsite(#loc114 at #loc181))
#loc215 = loc("l_i0"(#loc194))
#loc216 = loc(callsite(#loc134 at #loc181))
#loc218 = loc(callsite(#loc136 at #loc181))
#loc219 = loc(callsite(#loc137 at #loc181))
#loc220 = loc(callsite(#loc138 at #loc181))
#loc221 = loc(callsite(#loc139 at #loc181))
#loc222 = loc(callsite(#loc140 at #loc181))
#loc223 = loc(callsite(#loc141 at #loc181))
#loc225 = loc(callsite(#loc55 at #loc181))
#loc226 = loc(callsite(#loc56 at #loc181))
#loc227 = loc(callsite(#loc57 at #loc181))
#loc228 = loc(callsite(#loc143 at #loc181))
#loc229 = loc(callsite(#loc144 at #loc181))
#loc230 = loc(callsite(#loc145 at #loc181))
#loc231 = loc(callsite(#loc146 at #loc181))
#loc232 = loc(callsite(#loc147 at #loc181))
#loc233 = loc(callsite(#loc148 at #loc181))
#loc234 = loc(callsite(#loc149 at #loc181))
#loc235 = loc(callsite(#loc150 at #loc181))
#loc236 = loc(callsite(#loc151 at #loc181))
#loc237 = loc(callsite(#loc152 at #loc181))
#loc238 = loc(callsite(#loc134 at #loc198))
#loc240 = loc(callsite(#loc112 at #loc198))
#loc241 = loc(callsite(#loc136 at #loc198))
#loc242 = loc(callsite(#loc114 at #loc198))
#loc243 = loc(callsite(#loc137 at #loc198))
#loc244 = loc(callsite(#loc138 at #loc198))
#loc245 = loc(callsite(#loc139 at #loc198))
#loc246 = loc(callsite(#loc140 at #loc198))
#loc247 = loc(callsite(#loc141 at #loc198))
#loc249 = loc(callsite(#loc55 at #loc198))
#loc250 = loc(callsite(#loc56 at #loc198))
#loc251 = loc(callsite(#loc57 at #loc198))
#loc252 = loc(callsite(#loc143 at #loc198))
#loc253 = loc(callsite(#loc144 at #loc198))
#loc254 = loc(callsite(#loc145 at #loc198))
#loc255 = loc(callsite(#loc146 at #loc198))
#loc256 = loc(callsite(#loc147 at #loc198))
#loc257 = loc(callsite(#loc148 at #loc198))
#loc258 = loc(callsite(#loc149 at #loc198))
#loc259 = loc(callsite(#loc150 at #loc198))
#loc260 = loc(callsite(#loc151 at #loc198))
#loc261 = loc(callsite(#loc152 at #loc198))
#loc262 = loc("l_i1"(#loc215))
#loc263 = loc(callsite(#loc43 at #loc217))
#loc265 = loc(callsite(#loc52 at #loc224))
#loc267 = loc(callsite(#loc43 at #loc239))
#loc269 = loc(callsite(#loc52 at #loc248))
#loc271 = loc("m_i0"(#loc262))
#loc272 = loc(callsite(#loc45 at #loc263))
#loc273 = loc(callsite(#loc54 at #loc265))
#loc274 = loc(callsite(#loc45 at #loc267))
#loc275 = loc(callsite(#loc54 at #loc269))
#loc276 = loc("m_i1"(#loc271))
#loc277 = loc("offsetkv_y"(#loc276))
#loc278 = loc(callsite(#loc277 at #loc113))
