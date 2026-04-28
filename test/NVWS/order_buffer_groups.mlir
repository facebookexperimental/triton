// RUN: triton-opt %s --nvws-order-buffer-groups | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @orders_latest_ready_first
  tt.func @orders_latest_ready_first(
      %lb: i32, %ub: i32, %step: i32,
      %p_seed: tensor<128x128xf32, #linear>,
      %acc_seed: tensor<128x128xf32, #linear>,
      %rhs: !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>) {
    %true = arith.constant true
    // The accumulator is authored first, but its producer is shorter.
    // CHECK:      %[[P:.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
    // CHECK-NEXT: %[[ACC:.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32}
    %acc = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %p = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %acc_value = arith.addf %acc_seed, %acc_seed {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      ttng.tmem_store %acc_value, %acc, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %p_exp = math.exp2 %p_seed {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear>
      %p_value = tt.fp_to_fp %p_exp, rounding = rtne {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf8E5M2, #linear>
      ttng.tmem_store %p_value, %p, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x128xf8E5M2, #linear> -> !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %p, %rhs, %acc, %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "gemm", "computation"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @equal_readiness_is_stable
  tt.func @equal_readiness_is_stable(
      %lb: i32, %ub: i32, %step: i32,
      %x: tensor<128x128xf32, #linear>,
      %rhs: !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>) {
    %true = arith.constant true
    // CHECK:      ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 11 : i32}
    // CHECK-NEXT: ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32}
    %acc = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 11 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %lhs = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %acc_value = arith.addf %x, %x {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %acc_ready = ttg.convert_layout %acc_value {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %lhs_f32 = arith.addf %x, %x {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear>
      %lhs_value = tt.fp_to_fp %lhs_f32, rounding = rtne {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf8E5M2, #linear>
      ttng.tmem_store %acc_ready, %acc, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %lhs_value, %lhs, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x128xf8E5M2, #linear> -> !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %lhs, %rhs, %acc, %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E5M2, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "gemm", "computation"],
       ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
