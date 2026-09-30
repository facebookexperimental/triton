// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=400000 tma-store-pipelining=true" | FileCheck %s --check-prefix=ROOMY
// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=300000 tma-store-pipelining=true" | FileCheck %s --check-prefix=TRIMMED
// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" 2>/dev/null | FileCheck %s --check-prefix=FALLBACK
// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK

// The same-task TMEM A operand GEMM of ws_same_task_tmem.mlir. A is a pointer
// tl.load of a 128x128 f32 tile in the gemm partition, and the pipeliner that
// runs after warp specialization gives it num-stages - 1 = 2 cp.async buffers
// (128 KiB), plus up to a 32 KiB tile of layout-conversion scratch for the
// bf16 operand. The memory planner reserves those 160 KiB. With room to spare
// the TMA-loaded B operand gets three buffers; at a 300000-byte budget the
// reservation trims it to two; at the 232448-byte hardware limit even the
// planner's floors do not fit, so the kernel compiles without warp
// specialization rather than failing with OutOfResources later.
// ROOMY-LABEL: @tmem_operand_a_same_task
// ROOMY: ttg.local_alloc {{.*}} -> !ttg.memdesc<3x128x128xbf16
// ROOMY: ttg.warp_specialize

// TRIMMED-LABEL: @tmem_operand_a_same_task
// TRIMMED: ttg.local_alloc {{.*}} -> !ttg.memdesc<2x128x128xbf16
// TRIMMED: ttg.warp_specialize

// REMARK: remark: meta autoWS cannot fit its buffers plus 163840 bytes of pipelined loads and conversion scratch for an MMA operand written in the MMA's own partition in the 232448-byte shared-memory budget; compiling without warp specialization
// FALLBACK-LABEL: @tmem_operand_a_same_task
// FALLBACK-NOT: ttg.warp_specialize
// FALLBACK-NOT: async_task_id
// FALLBACK-NOT: ttg.partition =
// FALLBACK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xbf16
// FALLBACK: ttng.tc_gen5_mma
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_operand_a_same_task(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<5248> : tensor<128x1xi64, #blocked>
    %cst_0 = arith.constant dense<2048> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c5248_i32 = arith.constant 5248 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c41_i32 = arith.constant 41 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg2, [%c2048_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c5248_i32, %c1024_i32], [%c1024_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared1>
    %2 = tt.get_program_id x : i32
    scf.for %arg3 = %2 to %c128_i32 step %c148_i32  : i32 {
      %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
      %6 = arith.divsi %arg3, %c64_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %7 = arith.muli %6, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.subi %c16_i32, %7 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.minsi %8, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.remsi %arg3, %9 {ttg.partition = array<i32: 0>} : i32
      %11 = arith.addi %7, %10 {ttg.partition = array<i32: 0>} : i32
      %12 = arith.remsi %arg3, %c64_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.divsi %12, %9 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.muli %11, %c128_i32 {ttg.partition = array<i32: 0>} : i32
      %15 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.addi %16, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = arith.cmpi slt, %17, %cst_0 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
      %20 = tt.expand_dims %17 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %21 = arith.extsi %20 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
      %22 = arith.muli %21, %cst : tensor<128x1xi64, #blocked>
      %23 = tt.addptr %5, %22 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi64, #blocked>
      %24 = tt.broadcast %23 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %25 = tt.broadcast %19 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %26 = ttng.tmem_store %cst_2, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %27:2 = scf.for %arg4 = %c0_i32 to %c41_i32 step %c1_i32 iter_args(%arg5 = %false, %arg6 = %26) -> (i1, !ttg.async.token)  : i32 {
        %31 = arith.muli %arg4, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %32 = tt.descriptor_load %1[%31, %15] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xbf16, #shared1> -> tensor<128x128xbf16, #blocked1>
        %33 = ttg.local_alloc %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %34 = tt.splat %31 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %35 = arith.addi %34, %4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %36 = tt.expand_dims %35 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
        %37 = tt.broadcast %36 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %38 = tt.addptr %24, %37 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
        %39 = tt.load %38, %25, %cst_1 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
        %40 = arith.truncf %39 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %41 = ttg.convert_layout %40 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #linear>
        %result_5 = ttng.tmem_alloc %41 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>
        %42 = ttng.tc_gen5_mma %result_5, %33, %result[%arg6], %arg5, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %42 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_3, %token_4 = ttng.tmem_load %result[%27#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %28 = ttg.convert_layout %result_3 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      %29 = ttg.local_alloc %28 {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      %30 = ttng.async_tma_copy_local_to_global %0[%14, %15] %29 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %30   {ttg.partition = array<i32: 0>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
