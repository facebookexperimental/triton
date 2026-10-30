// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=1 capability=100 smem-budget=232448" | FileCheck %s

// A persistent FP32 (TF32) GEMM on sm100 whose A tile is a transposed TMA load
// (`a_desc.load([k, m]).T`) and whose B is K-major. tcgen05.mma cannot read a
// transposed FP32 operand from shared memory, so AccelerateMatmul stages A
// through registers into a transposed alloc whose memdesc_trans is a
// non-transposed operand. The TMA engine cannot write that alloc, so autoWS
// would leave the register copy in the MMA's own partition with no barrier
// against the asynchronous MMA still reading the previous tile (and
// swapTransposedLocalAllocs would turn the MMA operand back into a transposed
// one, which fails to lower). Warp specialization must bail out and leave the
// descriptor loads untouched.

// CHECK-LABEL: @fp32_transposed_a
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: nvws.descriptor_load
// CHECK: tt.descriptor_load
// CHECK: ttng.tc_gen5_mma

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @fp32_transposed_a(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg5, %arg3], [%0, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<32x128xf32, #shared>
    %2 = arith.extsi %arg5 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%2, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x32xf32, #shared>
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg3, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.muli %7, %5 : i32
    %9 = tt.get_program_id x : i32
    %10 = tt.get_num_programs x : i32
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg6 = %9 to %8 step %10  : i32 {
      %15 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %16 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %18 = arith.muli %16, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %19 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %20:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg8 = %false, %arg9 = %19) -> (i1, !ttg.async.token)  : i32 {
        %33 = tt.descriptor_load %1[%arg7, %17] {ttg.partition = array<i32: 1>} : !tt.tensordesc<32x128xf32, #shared> -> tensor<32x128xf32, #blocked>
        %34 = ttg.local_alloc %33 : (tensor<32x128xf32, #blocked>) -> !ttg.memdesc<32x128xf32, #shared1, #smem>
        %35 = ttg.memdesc_trans %34 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<32x128xf32, #shared1, #smem> -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %36 = tt.descriptor_load %3[%18, %arg7] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x32xf32, #shared> -> tensor<128x32xf32, #blocked1>
        %37 = ttg.local_alloc %36 {ttg.partition = array<i32: 1>} : (tensor<128x32xf32, #blocked1>) -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %38 = ttg.memdesc_trans %37 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem> -> !ttg.memdesc<32x128xf32, #shared1, #smem>
        %39 = ttng.tc_gen5_mma %35, %38, %result[%arg9], %arg8, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem>, !ttg.memdesc<32x128xf32, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %39 : i1, !ttg.async.token
      }
      %21 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addi %21, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.splat %18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %24 = arith.addi %23, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %25 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %26 = arith.muli %25, %13 : tensor<128x1xi32, #blocked>
      %27 = tt.addptr %14, %26 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
      %28 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %29 = tt.broadcast %27 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %30 = tt.broadcast %28 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %31 = tt.addptr %29, %30 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %32 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      tt.store %31, %32 {ttg.partition = array<i32: 2>} : tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// The same GEMM with K-major A and B: the TMA engine writes both operands in
// the layout the MMA reads, so it is still warp-specialized.

// CHECK-LABEL: @fp32_k_major
// CHECK: ttg.warp_specialize(
// CHECK: ttng.async_tma_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @fp32_k_major(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.extsi %arg5 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x32xf32, #shared>
    %2 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%0, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x32xf32, #shared>
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg3, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %4 : i32
    %8 = tt.get_program_id x : i32
    %9 = tt.get_num_programs x : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg6 = %8 to %7 step %9  : i32 {
      %14 = arith.divsi %arg6, %4 {ttg.partition = array<i32: 1>} : i32
      %15 = arith.remsi %arg6, %4 {ttg.partition = array<i32: 1>} : i32
      %16 = arith.muli %14, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %18 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %19:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg8 = %false, %arg9 = %18) -> (i1, !ttg.async.token)  : i32 {
        %32 = tt.descriptor_load %1[%16, %arg7] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x32xf32, #shared> -> tensor<128x32xf32, #blocked1>
        %33 = ttg.local_alloc %32 {ttg.partition = array<i32: 1>} : (tensor<128x32xf32, #blocked1>) -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %34 = tt.descriptor_load %2[%17, %arg7] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x32xf32, #shared> -> tensor<128x32xf32, #blocked1>
        %35 = ttg.local_alloc %34 {ttg.partition = array<i32: 1>} : (tensor<128x32xf32, #blocked1>) -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %36 = ttg.memdesc_trans %35 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem> -> !ttg.memdesc<32x128xf32, #shared1, #smem>
        %37 = ttng.tc_gen5_mma %33, %36, %result[%arg9], %arg8, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem>, !ttg.memdesc<32x128xf32, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %37 : i1, !ttg.async.token
      }
      %20 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = arith.addi %20, %10 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %23 = arith.addi %22, %11 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %24 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %25 = arith.muli %24, %12 : tensor<128x1xi32, #blocked>
      %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
      %27 = tt.expand_dims %23 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %28 = tt.broadcast %26 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %29 = tt.broadcast %27 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %30 = tt.addptr %28, %29 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %31 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      tt.store %30, %31 {ttg.partition = array<i32: 2>} : tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// A transposed bf16 A tile is read by the MMA as a transposed shared operand,
// which tcgen05.mma supports, so the TMA writes it directly and the GEMM is
// still warp-specialized.

// CHECK-LABEL: @bf16_transposed_a
// CHECK: ttg.warp_specialize(
// CHECK: ttng.async_tma_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @bf16_transposed_a(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg5, %arg3], [%0, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<32x128xbf16, #shared>
    %2 = arith.extsi %arg5 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%2, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared1>
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg3, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.muli %7, %5 : i32
    %9 = tt.get_program_id x : i32
    %10 = tt.get_num_programs x : i32
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg6 = %9 to %8 step %10  : i32 {
      %15 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %16 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %18 = arith.muli %16, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %19 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %20:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg8 = %false, %arg9 = %19) -> (i1, !ttg.async.token)  : i32 {
        %33 = tt.descriptor_load %1[%arg7, %17] {ttg.partition = array<i32: 1>} : !tt.tensordesc<32x128xbf16, #shared> -> tensor<32x128xbf16, #blocked1>
        %34 = ttg.local_alloc %33 {ttg.partition = array<i32: 1>} : (tensor<32x128xbf16, #blocked1>) -> !ttg.memdesc<32x128xbf16, #shared, #smem>
        %35 = ttg.memdesc_trans %34 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<32x128xbf16, #shared, #smem> -> !ttg.memdesc<128x32xbf16, #shared2, #smem>
        %36 = tt.descriptor_load %3[%18, %arg7] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared1> -> tensor<128x32xbf16, #blocked2>
        %37 = ttg.local_alloc %36 {ttg.partition = array<i32: 1>} : (tensor<128x32xbf16, #blocked2>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem>
        %38 = ttg.memdesc_trans %37 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xbf16, #shared1, #smem> -> !ttg.memdesc<32x128xbf16, #shared3, #smem>
        %39 = ttng.tc_gen5_mma %35, %38, %result[%arg9], %arg8, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xbf16, #shared2, #smem>, !ttg.memdesc<32x128xbf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %39 : i1, !ttg.async.token
      }
      %21 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addi %21, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.splat %18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %24 = arith.addi %23, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %25 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %26 = arith.muli %25, %13 : tensor<128x1xi32, #blocked>
      %27 = tt.addptr %14, %26 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
      %28 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %29 = tt.broadcast %27 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %30 = tt.broadcast %28 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %31 = tt.addptr %29, %30 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %32 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      tt.store %31, %32 {ttg.partition = array<i32: 2>} : tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
