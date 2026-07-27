// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// A Hopper dynamic-persistent GEMM combines WGMMA and its epilogue in the
// logical default partition. The inner loop shell must use that partition;
// its load and compute dependencies must not create an empty third partition.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @hopper_while_default_epilogue
  // CHECK: scf.while
  // CHECK: scf.for
  // CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 1>
  // CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: 1>
  // CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 1>
  // CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: 1>
  // CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: 0>
  // CHECK: ttng.warp_group_dot {{.*}}ttg.partition = array<i32: 0>
  // CHECK: scf.yield {{.*}}ttg.partition = array<i32: 0>
  // CHECK: } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 0>}
  // CHECK: arith.truncf {{.*}}ttg.partition = array<i32: 0>
  // CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: 0>
  // CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: 0>
  // CHECK: } attributes {
  // CHECK-SAME: tt.warp_specialize
  // CHECK-SAME: ttg.partition.types = ["epilogue", "load"]
  tt.func public @hopper_while_default_epilogue(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<128x64xf16, #shared>,
      %c_desc: !tt.tensordesc<128x128xf16, #shared>,
      %counter: !tt.ptr<i32>, %k_tiles: i32, %num_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %start = tt.get_program_id x : i32
    %result = scf.while (%tile = %start) : (i32) -> i32 {
      %valid = arith.cmpi slt, %tile, %num_tiles : i32
      scf.condition(%valid) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %inner = scf.for %ki = %c0 to %k_tiles step %c1
          iter_args(%acc = %acc_init) -> (tensor<128x128xf32, #mma>) : i32 {
        %a = tt.descriptor_load %a_desc[%tile, %ki] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%tile, %ki] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
        %dot = ttng.warp_group_dot %a_smem, %b_trans, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared_trans, #smem> -> tensor<128x128xf32, #mma>
        scf.yield %dot : tensor<128x128xf32, #mma>
      } {tt.scheduled_max_stage = 2 : i32}
      %out = arith.truncf %inner : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
      %out_blocked = ttg.convert_layout %out : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
      %out_smem = ttg.local_alloc %out_blocked : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %store = ttng.async_tma_copy_local_to_global %c_desc[%tile, %c0] %out_smem : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %store : !ttg.async.token
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %next : i32
    } attributes {tt.data_partition_factor = 1 : i32, tt.warp_specialize}
    tt.return
  }
}
