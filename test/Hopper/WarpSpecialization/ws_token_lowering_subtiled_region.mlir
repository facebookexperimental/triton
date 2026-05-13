// RUN: triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true" | FileCheck %s

// Test: token lowering correctly resolves inline NVWS ops inside
// SubtiledRegionOp tile bodies to hardware barrier ops when the
// SubtiledRegionOps are inside warp_specialize partition regions
// (FLATTEN=False persistent loop).

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @persistent_subtile_token_lowering
  //
  // SubtiledRegionOps with NVWS ops are inlined before token lowering.
  // Verify barrier ops appear in the warp_specialize partitions.
  // CHECK: ttg.warp_specialize
  //
  // Partition 0 (epilogue): inlined barrier + local_store ops
  // CHECK: partition0
  // CHECK: ttng.wait_barrier
  // CHECK: ttg.local_store
  // CHECK: ttng.arrive_barrier
  //
  // Partition 1 (store): inlined barrier + TMA copy ops
  // CHECK: partition1
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.arrive_barrier
  tt.func public @persistent_subtile_token_lowering(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %c_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>

    %start_pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_d = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_d = arith.divsi %num_pid_n, %c128_i32 : i32
    %k_tiles_a = arith.addi %K, %c63_i32 : i32
    %k_tiles = arith.divsi %k_tiles_a, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m_d, %num_pid_n_d : i32
    %tile_id_c_init = arith.subi %start_pid, %c148_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_d, %c8_i32 : i32

    // Persistent tile loop (not flattened)
    %tile_id_c_final = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32
        iter_args(%tile_id_c = %tile_id_c_init) -> (i32) : i32 {

      // Compute tile coordinates
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_d, %first_pid_m : i32
      %group_size_m_c = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_c : i32
      %pid_m_f = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_f = arith.divsi %pid_n, %group_size_m_c : i32
      %offs_am = arith.muli %pid_m_f, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_f, %c128_i32 : i32

      // TMEM alloc + MMA loop
      %accumulator, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_stored = ttng.tmem_store %cst, %accumulator[%acc_tok], %true {ttg.partition = array<i32: 1>}
          : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_result:2 = scf.for %ki = %c0_i32 to %k_tiles step %c1_i32
          iter_args(%use_acc = %false, %mma_tok = %acc_stored) -> (i1, !ttg.async.token) : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>}
            : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>}
            : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>}
            : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %mma = ttng.tc_gen5_mma %a_smem, %b_trans, %accumulator[%mma_tok], %use_acc, %true
            {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 0>}
            : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %mma : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}

      // Epilogue: next tile coordinates
      %tile_id_c_next = arith.addi %tile_id_c, %c148_i32 : i32
      %group_id2 = arith.divsi %tile_id_c_next, %num_pid_in_group : i32
      %first_pid_m2 = arith.muli %group_id2, %c8_i32 : i32
      %group_size_m2 = arith.subi %num_pid_m_d, %first_pid_m2 : i32
      %group_size_m2_c = arith.minsi %group_size_m2, %c8_i32 : i32
      %pid_m2 = arith.remsi %tile_id_c_next, %group_size_m2_c : i32
      %pid_m2_f = arith.addi %first_pid_m2, %pid_m2 : i32
      %pid_n2 = arith.remsi %tile_id_c_next, %num_pid_in_group : i32
      %pid_n2_f = arith.divsi %pid_n2, %group_size_m2_c : i32
      %offs_am_c = arith.muli %pid_m2_f, %c128_i32 : i32
      %offs_bn_c = arith.muli %pid_n2_f, %c128_i32 : i32

      // Epilogue: tmem_load → split → truncf → local_alloc → TMA store
      %loaded, %load_tok = ttng.tmem_load %accumulator[%mma_result#1] {ttg.partition = array<i32: 1>}
          : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %reshaped = tt.reshape %loaded {ttg.partition = array<i32: 1>}
          : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 1>}
          : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %lhs, %rhs = tt.split %transposed {ttg.partition = array<i32: 1>}
          : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>

      // Tile 0: truncf → local_alloc → TMA store
      %c0_trunc = arith.truncf %lhs {ttg.partition = array<i32: 1>}
          : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %c0_cvt = ttg.convert_layout %c0_trunc {ttg.partition = array<i32: 1>}
          : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      %c0_smem = ttg.local_alloc %c0_cvt {ttg.partition = array<i32: 1>}
          : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c] %c0_smem {ttg.partition = array<i32: 2>}
          : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {ttg.partition = array<i32: 2>} : !ttg.async.token

      // Tile 1: truncf → local_alloc → TMA store
      %c1_trunc = arith.truncf %rhs {ttg.partition = array<i32: 1>}
          : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %c1_cvt = ttg.convert_layout %c1_trunc {ttg.partition = array<i32: 1>}
          : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      %offs_bn_c2 = arith.addi %offs_bn_c, %c64_i32 : i32
      %c1_smem = ttg.local_alloc %c1_cvt {ttg.partition = array<i32: 1>}
          : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c2] %c1_smem {ttg.partition = array<i32: 2>}
          : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 {ttg.partition = array<i32: 2>} : !ttg.async.token

      scf.yield %tile_id_c_next : i32
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.smem_alloc_algo = 0 : i32,
       tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["gemm", "epilogue", "epilogue_store", "load", "computation"],
       ttg.warp_specialize.tag = 0 : i32}

    tt.return
  }
}
