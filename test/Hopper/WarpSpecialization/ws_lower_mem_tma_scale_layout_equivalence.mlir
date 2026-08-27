// RUN: triton-opt %s --nvgpu-warp-specialization="num-stages=4 capability=100 smem-budget=232448" --verify-each | FileCheck %s
// RUN: triton-opt %s --nvgpu-warp-specialization="num-stages=4 capability=100 smem-budget=232448" --verify-each | FileCheck %s --check-prefix=REUSE

// MXFP8 scales are allocated with a shared_linear layout -- what the SMEM->TMEM
// copy needs. It describes the same layout as the descriptor's nvmma_shared,
// but is a different attribute, and TMA can only write nvmma_shared. Rather
// than give up on fusing the load into that buffer (which costs an SMEM ->
// register -> SMEM round trip through local_load), re-spell the fused buffer
// with the descriptor's layout: same bits, so TMA writes the scale buffer
// directly and the consumer chain is unchanged.

// CHECK-DAG: #[[$SCALE:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>

// CHECK-LABEL: @scaled_mm_autows_kernel

// Both scale loads must land in an nvmma_shared buffer, not shared_linear.
// CHECK: ttng.async_tma_copy_global_to_local{{.*}}!tt.tensordesc<1x1x1x2x256xui8, #[[$SCALE]]>{{.*}}-> !ttg.memdesc<1x1x1x2x256xi8, #[[$SCALE]], #smem, mutable>
// CHECK: ttng.async_tma_copy_global_to_local{{.*}}!tt.tensordesc<1x2x1x2x256xui8, #[[$SCALE]]>{{.*}}-> !ttg.memdesc<1x2x1x2x256xi8, #[[$SCALE]], #smem, mutable>

// ...and the buffer must be reused, not staged through registers. Declining
// the fusion still satisfies the checks above -- the load just gets its own
// nvmma_shared buffer -- but leaves one local_load per scale behind, so the
// absence of any local_load is what separates reuse from fall-through. This
// runs as its own prefix with no other REUSE directives, so the CHECK-NOT
// spans the whole module rather than a region between two matches.
// REUSE-NOT: ttg.local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 2, 2], order = [4, 3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 8], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 2, 1, 2, 1], order = [4, 3, 2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared3 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8], [0, 0, 0, 0, 16], [0, 0, 0, 0, 32], [0, 0, 0, 0, 64], [0, 0, 0, 0, 128], [0, 0, 0, 1, 0]]}, alignment = 128>
#shared4 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]]}, alignment = 128>
#shared5 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]]}, alignment = 128>
#shared6 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]}, alignment = 128>
#shared7 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8], [0, 0, 0, 0, 16], [0, 0, 0, 0, 32], [0, 0, 0, 0, 64], [0, 0, 0, 0, 128], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]}, alignment = 128>
#shared8 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared9 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared10 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [128, 0]]}, alignment = 128>
#shared11 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scaled_mm_autows_kernel(%a_desc: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %b_desc: !tt.tensordesc<256x128xf8E4M3FN, #shared>, %c_desc: !tt.tensordesc<128x256xbf16, #shared1>, %scale_a_ptr: !tt.tensordesc<1x1x1x2x256xui8, #shared2>, %scale_b_ptr: !tt.tensordesc<1x2x1x2x256xui8, #shared2>) attributes {noinline = false} {
    %false = arith.constant false
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear>
    %tile_id_c_4 = scf.for %tile_id = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%tile_id_c_5 = %c0_i32) -> (i32)  : i32 {
      %accumulator, %accumulator_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_10 = ttng.tmem_store %cst, %accumulator[%accumulator_9], %true {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #linear> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_11:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%accumulator_32 = %false, %accumulator_33 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c128_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        %a_tile = tt.descriptor_load %a_desc[%c0_i32, %offs_k] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked>
        %a_tile_34 = ttg.local_alloc %a_tile {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 3>} : (tensor<128x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %b_tile = tt.descriptor_load %b_desc[%c0_i32, %offs_k] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<256x128xf8E4M3FN, #shared> -> tensor<256x128xf8E4M3FN, #blocked>
        %sa_packed = tt.descriptor_load %scale_a_ptr[%c0_i32, %c0_i32, %ki, %c0_i32, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<1x1x1x2x256xui8, #shared2> -> tensor<1x1x1x2x256xi8, #blocked1>
        %sb_packed = tt.descriptor_load %scale_b_ptr[%c0_i32, %c0_i32, %ki, %c0_i32, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<1x2x1x2x256xui8, #shared2> -> tensor<1x2x1x2x256xi8, #blocked2>
        %sa = ttg.local_alloc %sa_packed {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<1x1x1x2x256xi8, #blocked1>) -> !ttg.memdesc<1x1x1x2x256xi8, #shared3, #smem>
        %sa_35 = ttg.memdesc_reshape %sa {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1x1x2x256xi8, #shared3, #smem> -> !ttg.memdesc<1x1x32x4x4xi8, #shared4, #smem>
        %sa_36 = ttg.memdesc_trans %sa_35 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1x32x4x4xi8, #shared4, #smem> -> !ttg.memdesc<1x4x32x1x4xi8, #shared5, #smem>
        %sa_37 = ttg.memdesc_reshape %sa_36 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x4x32x1x4xi8, #shared5, #smem> -> !ttg.memdesc<128x4xi8, #shared6, #smem>
        %sb = ttg.local_alloc %sb_packed {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<1x2x1x2x256xi8, #blocked2>) -> !ttg.memdesc<1x2x1x2x256xi8, #shared7, #smem>
        %sb_38 = ttg.memdesc_reshape %sb {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x2x1x2x256xi8, #shared7, #smem> -> !ttg.memdesc<2x1x32x4x4xi8, #shared8, #smem>
        %sb_39 = ttg.memdesc_trans %sb_38 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x1x32x4x4xi8, #shared8, #smem> -> !ttg.memdesc<2x4x32x1x4xi8, #shared9, #smem>
        %sb_40 = ttg.memdesc_reshape %sb_39 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x4x32x1x4xi8, #shared9, #smem> -> !ttg.memdesc<256x4xi8, #shared10, #smem>
        %accumulator_41 = ttg.local_alloc %b_tile {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 3>} : (tensor<256x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
        %accumulator_42 = ttg.memdesc_trans %accumulator_41 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x256xf8E4M3FN, #shared11, #smem>
        %accumulator_43 = ttng.tc_gen5_mma_scaled %a_tile_34, %accumulator_42, %accumulator[%accumulator_33], %sa_37, %sb_40, %accumulator_32, %true lhs = e4m3 rhs = e4m3 {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared11, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #shared6, #smem>, !ttg.memdesc<256x4xi8, #shared10, #smem>
        scf.yield %true, %accumulator_43 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 3 : i32}
      %tile_id_c_12 = arith.addi %tile_id_c_5, %c1_i32 : i32
      %accumulator_21, %accumulator_22 = ttng.tmem_load %accumulator[%accumulator_11#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %o = arith.truncf %accumulator_21 {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #linear> to tensor<128x256xbf16, #linear>
      %o_st = ttg.local_alloc %o {ttg.partition = array<i32: 0>} : (tensor<128x256xbf16, #linear>) -> !ttg.memdesc<128x256xbf16, #shared1, #smem, mutable>
      %o_tok = ttng.async_tma_copy_local_to_global %c_desc[%c0_i32, %c0_i32] %o_st {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x256xbf16, #shared1>, !ttg.memdesc<128x256xbf16, #shared1, #smem, mutable> -> !ttg.async.token
      nvws.tma_store_wait %o_st {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x256xbf16, #shared1, #smem, mutable>
      scf.yield %tile_id_c_12 : i32
    } {tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
