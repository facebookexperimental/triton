// RUN: triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=2 smem-budget=232448 tma-store-pipelining=true" | FileCheck %s --implicit-check-not=tt.multicast_axes

// CHECK-LABEL: @surviving_while_multicast
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: scf.while {{.*}} : (i32, i1, i64, i64) -> (i32, i64, i64)
// CHECK: scf.if
// CHECK: partition0
// CHECK: scf.while {{.*}} : (i32, i1, i64, i64) -> (i32, i64, i64)
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: scf.if
// CHECK: ttg.warp_return
// CHECK: partition1
// CHECK: scf.while {{.*}} : (i32, i1, i64, i64) -> (i32, i64, i64)
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK-NOT: scf.if
// CHECK: ttg.warp_return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  ttg.early_tma_store_lowering = true,
  ttg.min_reg_auto_ws = 24 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100",
  "ttg.threads-per-warp" = 32 : i32,
  "ttng.two-ctas" = false
} {
  tt.func public @surviving_while_multicast(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<128x64xf16, #shared>,
      %c_desc: !tt.tensordesc<128x128xf16, #shared>,
      %m: i32, %n: i32, %k: i32) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %m_rounded = arith.addi %m, %c127_i32 : i32
    %num_m = arith.divsi %m_rounded, %c128_i32 : i32
    %n_rounded = arith.addi %n, %c127_i32 : i32
    %num_n = arith.divsi %n_rounded, %c128_i32 : i32
    %k_rounded = arith.addi %k, %c63_i32 : i32
    %num_k = arith.divsi %k_rounded, %c64_i32 : i32
    %pid_m = tt.get_program_id x : i32
    %local_m = arith.remsi %pid_m, %c2_i32 : i32
    %pid_n = tt.get_program_id y : i32
    %local_n = arith.remsi %pid_n, %c2_i32 : i32
    %cluster_m = arith.divsi %pid_m, %c2_i32 : i32
    %cluster_n = arith.divsi %pid_n, %c2_i32 : i32
    %num_n_plus = arith.addi %num_n, %c1_i32 : i32
    %num_cluster_n = arith.divsi %num_n_plus, %c2_i32 : i32
    %cluster_base = arith.muli %cluster_m, %num_cluster_n : i32
    %tile_id = arith.addi %cluster_base, %cluster_n : i32
    %num_m_plus = arith.addi %num_m, %c1_i32 : i32
    %num_cluster_m = arith.divsi %num_m_plus, %c2_i32 : i32
    %num_tiles = arith.muli %num_cluster_m, %num_cluster_n : i32
    %grid_m = tt.get_num_programs x : i32
    %clusters_m = arith.divsi %grid_m, %c2_i32 : i32
    %grid_n = tt.get_num_programs y : i32
    %clusters_n = arith.divsi %grid_n, %c2_i32 : i32
    %tile_stride = arith.muli %clusters_m, %clusters_n : i32
    %valid = arith.cmpi slt, %tile_id, %num_tiles : i32
    %result = scf.while (%tile = %tile_id, %keep_going = %valid) : (i32, i1) -> i32 {
      scf.condition(%keep_going) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %tile_m = arith.divsi %tile, %num_cluster_n : i32
      %macro_m = arith.muli %tile_m, %c2_i32 : i32
      %out_m = arith.addi %macro_m, %local_m : i32
      %tile_n = arith.remsi %tile, %num_cluster_n : i32
      %macro_n = arith.muli %tile_n, %c2_i32 : i32
      %out_n = arith.addi %macro_n, %local_n : i32
      %offs_m = arith.muli %out_m, %c128_i32 : i32
      %offs_n = arith.muli %out_n, %c128_i32 : i32
      %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %init = ttng.tmem_store %zero, %acc[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %loop:2 = scf.for %ki = %c0_i32 to %num_k step %c1_i32 iter_args(%used = %false, %acc_token = %init) -> (i1, !ttg.async.token) : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_m, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 1>, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_n, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 0>, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %b_alloc = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b_trans = ttg.memdesc_trans %b_alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %next_token = ttng.tc_gen5_mma %a_alloc, %b_trans, %acc[%acc_token], %used, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %next_token : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %loaded, %load_token = ttng.tmem_load %acc[%loop#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %valid_m = arith.cmpi slt, %out_m, %num_m : i32
      %valid_n = arith.cmpi slt, %out_n, %num_n : i32
      %store = arith.andi %valid_m, %valid_n : i1
      scf.if %store {
        %narrow = arith.truncf %loaded {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %layout = ttg.convert_layout %narrow {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
        %staging = ttg.local_alloc %layout {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %store_token = ttng.async_tma_copy_local_to_global %c_desc[%offs_m, %offs_n] %staging {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %store_token {ttg.partition = array<i32: 0>} : !ttg.async.token
      } {ttg.partition = array<i32: 0>}
      %next_tile = arith.addi %tile, %tile_stride : i32
      %next_valid = arith.cmpi slt, %next_tile, %num_tiles : i32
      scf.yield %next_tile, %next_valid : i32, i1
    } attributes {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
