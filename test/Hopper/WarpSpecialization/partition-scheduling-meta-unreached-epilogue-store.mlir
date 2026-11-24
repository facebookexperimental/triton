// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta="separate-epilogue-store" | FileCheck %s

// An epilogue store whose value does not depend on the accumulator (x * 2
// loaded through a pointer) is not reachable from any partition, so partition
// propagation used to leave it unscheduled. It must be scheduled with the
// accumulator store, both when it has its own mask and when it shares the
// accumulator store's mask.

// CHECK-LABEL: @side_store_own_mask
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P:[0-9]+]]>}
// CHECK: tt.load
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P]]>}
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @side_store_own_mask(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<131072> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
    %cst_2 = arith.constant dense<1024> : tensor<128x1xi32, #blocked>
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<128x128xbf16, #blocked>
    %true = arith.constant true
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c2048_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_1 : tensor<1x128xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %8 = tt.broadcast %5 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %11 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    scf.for %arg5 = %2 to %c8_i32 step %c148_i32  : i32 {
      %12 = arith.muli %arg5, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %13 = ttng.tmem_store %cst_4, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %14:2 = scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %13) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg6, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %0[%12, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %35 = tt.descriptor_load %1[%32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %37 = ttng.tc_gen5_mma %34, %36, %result[%arg8], %arg7, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %37 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %15 = tt.splat %12 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = arith.addi %15, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %18 = arith.cmpi slt, %17, %cst_2 : tensor<128x1xi32, #blocked>
      %19 = tt.broadcast %18 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %20 = arith.andi %19, %7 : tensor<128x128xi1, #blocked>
      %21 = arith.muli %17, %cst_0 : tensor<128x1xi32, #blocked>
      %22 = tt.broadcast %21 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %23 = arith.addi %22, %8 : tensor<128x128xi32, #blocked>
      %24 = tt.addptr %9, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %result_5, %token_6 = ttng.tmem_load %result[%14#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %25 = arith.truncf %result_5 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %26 = ttg.convert_layout %25 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      tt.store %24, %26, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %27 = tt.addptr %10, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %28 = tt.load %27, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %29 = tt.addptr %11, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %30 = arith.mulf %28, %cst_3 : tensor<128x128xbf16, #blocked>
      %31 = arith.cmpi slt, %23, %cst : tensor<128x128xi32, #blocked>
      tt.store %29, %30, %31 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}


// -----

// CHECK-LABEL: @side_store_shared_mask
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P:[0-9]+]]>}
// CHECK: tt.load
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P]]>}
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @side_store_shared_mask(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<131072> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
    %cst_2 = arith.constant dense<1024> : tensor<128x1xi32, #blocked>
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<128x128xbf16, #blocked>
    %true = arith.constant true
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c2048_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_1 : tensor<1x128xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %8 = tt.broadcast %5 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %11 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    scf.for %arg5 = %2 to %c8_i32 step %c148_i32  : i32 {
      %12 = arith.muli %arg5, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %13 = ttng.tmem_store %cst_4, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %14:2 = scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %13) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg6, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %0[%12, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %35 = tt.descriptor_load %1[%32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %37 = ttng.tc_gen5_mma %34, %36, %result[%arg8], %arg7, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %37 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %15 = tt.splat %12 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = arith.addi %15, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %18 = arith.cmpi slt, %17, %cst_2 : tensor<128x1xi32, #blocked>
      %19 = tt.broadcast %18 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %20 = arith.andi %19, %7 : tensor<128x128xi1, #blocked>
      %21 = arith.muli %17, %cst_0 : tensor<128x1xi32, #blocked>
      %22 = tt.broadcast %21 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %23 = arith.addi %22, %8 : tensor<128x128xi32, #blocked>
      %24 = tt.addptr %9, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %result_5, %token_6 = ttng.tmem_load %result[%14#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %25 = arith.truncf %result_5 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %26 = ttg.convert_layout %25 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      tt.store %24, %26, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %27 = tt.addptr %10, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %28 = tt.load %27, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %29 = tt.addptr %11, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %30 = arith.mulf %28, %cst_3 : tensor<128x128xbf16, #blocked>
      tt.store %29, %30, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store unreachable from every partition is still scheduled.
// CHECK-LABEL: @scalar_side_store
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i32>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_side_store(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xf32, #shared1>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %owner_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c3_i32 = arith.constant 3 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %start_pid = tt.get_program_id x : i32
    %grid_m = arith.addi %M, %c127_i32 : i32
    %grid_m_0 = arith.divsi %grid_m, %c128_i32 : i32
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_1 = arith.divsi %grid_n, %c128_i32 : i32
    %num_tiles = arith.muli %grid_m_0, %grid_n_1 : i32
    %num_pid_in_group = arith.muli %grid_n_1, %c8_i32 : i32
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %grid_m_0, %first_pid_m : i32
      %group_size_m_2 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_2 : i32
      %pid_m_3 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_4 = arith.divsi %pid_n, %group_size_m_2 : i32
      %offs_am = arith.muli %pid_m_3, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_4, %c128_i32 : i32
      %acc, %acc_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_6 = ttng.tmem_store %cst, %acc[%acc_5], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_7:2 = scf.for %ki = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc_10 = %false, %acc_11 = %acc_6) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %a_12 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_13 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_14 = ttg.memdesc_trans %acc_13 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
        %acc_15 = ttng.tc_gen5_mma %a_12, %acc_14, %acc[%acc_11], %acc_10, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_15 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_8, %acc_9 = ttng.tmem_load %acc[%acc_7#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %0 = ttg.convert_layout %acc_8 : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      %c_desc_staging = ttg.local_alloc %0 : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %1 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am, %offs_bn] %c_desc_staging : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %1   : !ttg.async.token
      %2 = tt.addptr %owner_ptr, %tile_id : !tt.ptr<i32>, i32
      %3 = arith.muli %tile_id, %c3_i32 : i32
      %4 = arith.addi %3, %c1_i32 : i32
      tt.store %2, %4 : !tt.ptr<i32>
    } {tt.data_partition_factor = 1 : i32, tt.warp_specialize}
    tt.return
  }
}
