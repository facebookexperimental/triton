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

// A scalar side store in the persistent tile loop whose value does not depend
// on any load or the accumulator (here tile_id * 3 + 1) is also unreachable
// from every partition. Left unscheduled, warp specialization dropped it.
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



// -----

// A scalar store of an inline_asm %smid (non-pure, no operands) in a persistent tl.range loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode2_static
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode2_static(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %num_tiles = arith.addi %M, %c127_i32 : i32
    %num_tiles_1 = arith.divsi %num_tiles, %c128_i32 : i32
    %num_tiles_2 = arith.muli %num_tiles_1, %grid_n_0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    scf.for %tile = %0 to %num_tiles_2 step %1  : i32 {
      %smid = tt.elementwise_inline_asm "mov.u32 $0, %smid;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
      %2 = tt.addptr %tr, %tile : !tt.ptr<i64>, i32
      %3 = arith.extsi %smid : i32 to i64
      tt.store %2, %3 : !tt.ptr<i64>
      %pm = arith.divsi %tile, %grid_n_0 : i32
      %pn = arith.remsi %tile, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_3 = arith.muli %pn, %c128_i32 : i32
      %acc_4, %acc_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_6 = ttng.tmem_store %cst, %acc_4[%acc_5], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_7:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_10 = %false, %acc_11 = %acc_6) -> (i1, !ttg.async.token)  : i32 {
        %acc_12 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_13 = tt.descriptor_load %a_desc[%acc, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_14 = ttg.local_alloc %acc_13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_15 = tt.descriptor_load %b_desc[%acc_3, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_16 = ttg.local_alloc %acc_15 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_17 = ttg.memdesc_trans %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_18 = ttng.tc_gen5_mma %acc_14, %acc_17, %acc_4[%acc_11], %acc_10, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_18 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_8, %acc_9 = ttng.tmem_load %acc_4[%acc_7#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %4 = arith.truncf %acc_8 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %5 = ttg.convert_layout %4 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %5 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %6 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_3] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %6   : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store of an inline_asm %smid (non-pure, no operands) in a CLC while loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode2_clc
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode2_clc(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %sched = tt.get_program_id x : i32
    %sched._z = scf.while (%sched._valid = %true, %sched._x = %sched) : (i1, i32) -> i32 {
      scf.condition(%sched._valid) %sched._x : i32
    } do {
    ^bb0(%sched._x: i32):
      %sched_1 = ttng.clc_try_cancel_async : !ttg.async.token
      %smid = tt.elementwise_inline_asm "mov.u32 $0, %smid;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
      %0 = tt.addptr %tr, %sched._x : !tt.ptr<i64>, i32
      %1 = arith.extsi %smid : i32 to i64
      tt.store %0, %1 : !tt.ptr<i64>
      %pm = arith.divsi %sched._x, %grid_n_0 : i32
      %pn = arith.remsi %sched._x, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_2 = arith.muli %pn, %c128_i32 : i32
      %acc_3, %acc_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_5 = ttng.tmem_store %cst, %acc_3[%acc_4], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_6:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_13 = %false, %acc_14 = %acc_5) -> (i1, !ttg.async.token)  : i32 {
        %acc_15 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_16 = tt.descriptor_load %a_desc[%acc, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_17 = ttg.local_alloc %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_18 = tt.descriptor_load %b_desc[%acc_2, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_19 = ttg.local_alloc %acc_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_20 = ttg.memdesc_trans %acc_19 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_21 = ttng.tc_gen5_mma %acc_17, %acc_20, %acc_3[%acc_14], %acc_13, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_21 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_7, %acc_8 = ttng.tmem_load %acc_3[%acc_6#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %2 = arith.truncf %acc_7 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %3 = ttg.convert_layout %2 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %3 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %4 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_2] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %4   : !ttg.async.token
      %sched_9, %sched_10, %sched_11, %sched_12 = ttng.clc_read %sched_1 : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %sched_9, %sched_10 : i1, i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store of program_id(0) + 1 in a persistent tl.range loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode4_static
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode4_static(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %num_tiles = arith.addi %M, %c127_i32 : i32
    %num_tiles_1 = arith.divsi %num_tiles, %c128_i32 : i32
    %num_tiles_2 = arith.muli %num_tiles_1, %grid_n_0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %0, %c1_i32 : i32
    %3 = arith.extsi %2 : i32 to i64
    scf.for %tile = %0 to %num_tiles_2 step %1  : i32 {
      %4 = tt.addptr %tr, %tile : !tt.ptr<i64>, i32
      tt.store %4, %3 : !tt.ptr<i64>
      %pm = arith.divsi %tile, %grid_n_0 : i32
      %pn = arith.remsi %tile, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_3 = arith.muli %pn, %c128_i32 : i32
      %acc_4, %acc_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_6 = ttng.tmem_store %cst, %acc_4[%acc_5], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_7:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_10 = %false, %acc_11 = %acc_6) -> (i1, !ttg.async.token)  : i32 {
        %acc_12 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_13 = tt.descriptor_load %a_desc[%acc, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_14 = ttg.local_alloc %acc_13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_15 = tt.descriptor_load %b_desc[%acc_3, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_16 = ttg.local_alloc %acc_15 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_17 = ttg.memdesc_trans %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_18 = ttng.tc_gen5_mma %acc_14, %acc_17, %acc_4[%acc_11], %acc_10, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_18 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_8, %acc_9 = ttng.tmem_load %acc_4[%acc_7#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %5 = arith.truncf %acc_8 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %6 = ttg.convert_layout %5 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %6 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %7 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_3] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %7   : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store of program_id(0) + 1 in a CLC while loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode4_clc
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode4_clc(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %sched = tt.get_program_id x : i32
    %0 = arith.addi %sched, %c1_i32 : i32
    %1 = arith.extsi %0 : i32 to i64
    %sched._z = scf.while (%sched._valid = %true, %sched._x = %sched) : (i1, i32) -> i32 {
      scf.condition(%sched._valid) %sched._x : i32
    } do {
    ^bb0(%sched._x: i32):
      %sched_1 = ttng.clc_try_cancel_async : !ttg.async.token
      %2 = tt.addptr %tr, %sched._x : !tt.ptr<i64>, i32
      tt.store %2, %1 : !tt.ptr<i64>
      %pm = arith.divsi %sched._x, %grid_n_0 : i32
      %pn = arith.remsi %sched._x, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_2 = arith.muli %pn, %c128_i32 : i32
      %acc_3, %acc_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_5 = ttng.tmem_store %cst, %acc_3[%acc_4], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_6:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_13 = %false, %acc_14 = %acc_5) -> (i1, !ttg.async.token)  : i32 {
        %acc_15 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_16 = tt.descriptor_load %a_desc[%acc, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_17 = ttg.local_alloc %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_18 = tt.descriptor_load %b_desc[%acc_2, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_19 = ttg.local_alloc %acc_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_20 = ttg.memdesc_trans %acc_19 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_21 = ttng.tc_gen5_mma %acc_17, %acc_20, %acc_3[%acc_14], %acc_13, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_21 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_7, %acc_8 = ttng.tmem_load %acc_3[%acc_6#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %3 = arith.truncf %acc_7 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %4 = ttg.convert_layout %3 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %4 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %5 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_2] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %5   : !ttg.async.token
      %sched_9, %sched_10, %sched_11, %sched_12 = ttng.clc_read %sched_1 : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %sched_9, %sched_10 : i1, i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store of three i64 values computed from the tile id in a persistent tl.range loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode7_static
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode7_static(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c3_i32 = arith.constant 3 : i32
    %c32_i64 = arith.constant 32 : i64
    %c7_i64 = arith.constant 7 : i64
    %c1_i32 = arith.constant 1 : i32
    %c5_i64 = arith.constant 5 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %num_tiles = arith.addi %M, %c127_i32 : i32
    %num_tiles_1 = arith.divsi %num_tiles, %c128_i32 : i32
    %num_tiles_2 = arith.muli %num_tiles_1, %grid_n_0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    scf.for %tile = %0 to %num_tiles_2 step %1  : i32 {
      %t64 = arith.extsi %tile : i32 to i64
      %2 = arith.muli %tile, %c3_i32 : i32
      %3 = tt.addptr %tr, %2 : !tt.ptr<i64>, i32
      %4 = arith.shli %t64, %c32_i64 : i64
      %5 = arith.ori %4, %c7_i64 : i64
      tt.store %3, %5 : !tt.ptr<i64>
      %6 = tt.addptr %3, %c1_i32 : !tt.ptr<i64>, i32
      %7 = arith.addi %t64, %c5_i64 : i64
      tt.store %6, %7 : !tt.ptr<i64>
      %8 = tt.addptr %3, %c2_i32 : !tt.ptr<i64>, i32
      tt.store %8, %t64 : !tt.ptr<i64>
      %pm = arith.divsi %tile, %grid_n_0 : i32
      %pn = arith.remsi %tile, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_3 = arith.muli %pn, %c128_i32 : i32
      %acc_4, %acc_5 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_6 = ttng.tmem_store %cst, %acc_4[%acc_5], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_7:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_10 = %false, %acc_11 = %acc_6) -> (i1, !ttg.async.token)  : i32 {
        %acc_12 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_13 = tt.descriptor_load %a_desc[%acc, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_14 = ttg.local_alloc %acc_13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_15 = tt.descriptor_load %b_desc[%acc_3, %acc_12] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_16 = ttg.local_alloc %acc_15 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_17 = ttg.memdesc_trans %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_18 = ttng.tc_gen5_mma %acc_14, %acc_17, %acc_4[%acc_11], %acc_10, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_18 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_8, %acc_9 = ttng.tmem_load %acc_4[%acc_7#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %9 = arith.truncf %acc_8 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %10 = ttg.convert_layout %9 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %10 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %11 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_3] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %11   : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// A scalar store of three i64 values computed from the tile id in a CLC while loop (from clc-steal's repro): the
// value comes from an op whose operands are loop invariant or absent, so no
// partition reaches it and it used to be dropped.
// CHECK-LABEL: @scalar_store_mode7_clc
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
// CHECK: tt.store %{{.*}}, %{{.*}} {ttg.partition = array<i32: {{[0-9]+}}>} : !tt.ptr<i64>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @scalar_store_mode7_clc(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %a_desc.shape.0: i32, %a_desc.shape.1: i32, %a_desc.stride.0: i64, %a_desc.stride.1: i64, %b_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc.shape.0: i32, %b_desc.shape.1: i32, %b_desc.stride.0: i64, %b_desc.stride.1: i64, %c_desc: !tt.tensordesc<128x128xbf16, #shared>, %c_desc.shape.0: i32, %c_desc.shape.1: i32, %c_desc.stride.0: i64, %c_desc.stride.1: i64, %tr: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %true = arith.constant true
    %c3_i32 = arith.constant 3 : i32
    %c32_i64 = arith.constant 32 : i64
    %c7_i64 = arith.constant 7 : i64
    %c1_i32 = arith.constant 1 : i32
    %c5_i64 = arith.constant 5 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %grid_n = arith.addi %N, %c127_i32 : i32
    %grid_n_0 = arith.divsi %grid_n, %c128_i32 : i32
    %sched = tt.get_program_id x : i32
    %sched._z = scf.while (%sched._valid = %true, %sched._x = %sched) : (i1, i32) -> i32 {
      scf.condition(%sched._valid) %sched._x : i32
    } do {
    ^bb0(%sched._x: i32):
      %sched_1 = ttng.clc_try_cancel_async : !ttg.async.token
      %t64 = arith.extsi %sched._x : i32 to i64
      %0 = arith.muli %sched._x, %c3_i32 : i32
      %1 = tt.addptr %tr, %0 : !tt.ptr<i64>, i32
      %2 = arith.shli %t64, %c32_i64 : i64
      %3 = arith.ori %2, %c7_i64 : i64
      tt.store %1, %3 : !tt.ptr<i64>
      %4 = tt.addptr %1, %c1_i32 : !tt.ptr<i64>, i32
      %5 = arith.addi %t64, %c5_i64 : i64
      tt.store %4, %5 : !tt.ptr<i64>
      %6 = tt.addptr %1, %c2_i32 : !tt.ptr<i64>, i32
      tt.store %6, %t64 : !tt.ptr<i64>
      %pm = arith.divsi %sched._x, %grid_n_0 : i32
      %pn = arith.remsi %sched._x, %grid_n_0 : i32
      %acc = arith.muli %pm, %c128_i32 : i32
      %acc_2 = arith.muli %pn, %c128_i32 : i32
      %acc_3, %acc_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_5 = ttng.tmem_store %cst, %acc_3[%acc_4], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_6:2 = scf.for %ki = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_13 = %false, %acc_14 = %acc_5) -> (i1, !ttg.async.token)  : i32 {
        %acc_15 = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %acc_16 = tt.descriptor_load %a_desc[%acc, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_17 = ttg.local_alloc %acc_16 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_18 = tt.descriptor_load %b_desc[%acc_2, %acc_15] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
        %acc_19 = ttg.local_alloc %acc_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %acc_20 = ttg.memdesc_trans %acc_19 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared1, #smem>
        %acc_21 = ttng.tc_gen5_mma %acc_17, %acc_20, %acc_3[%acc_14], %acc_13, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %acc_21 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %acc_7, %acc_8 = ttng.tmem_load %acc_3[%acc_6#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %7 = arith.truncf %acc_7 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %8 = ttg.convert_layout %7 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %c_desc_staging = ttg.local_alloc %8 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %9 = ttng.async_tma_copy_local_to_global %c_desc[%acc, %acc_2] %c_desc_staging : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %9   : !ttg.async.token
      %sched_9, %sched_10, %sched_11, %sched_12 = ttng.clc_read %sched_1 : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %sched_9, %sched_10 : i1, i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}
