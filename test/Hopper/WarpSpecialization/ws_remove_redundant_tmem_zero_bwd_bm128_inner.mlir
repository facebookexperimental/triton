// RUN: triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=2 smem-budget=232448" | FileCheck %s
// CHECK-LABEL: tt.func public @_attn_bwd
// The MMA's parent loop is proved non-empty by tl.assume(hi > lo) on its bounds,
// which is what lets removeRedundantTmemZeroStores drop the redundant operand-D
// zero-store.
// Dead load-task rematerialization clones of m/Di must be removed before
// descriptor conversion derives the local_load consumer set. Otherwise these
// loads carry tasks {0, 3} and create an unsafe self-consumer TMA channel.
// CHECK-COUNT-2: ttg.local_load {{.*}}async_task_id = array<i32: 0>{{.*}}!ttg.memdesc<128xf32
// CHECK-NOT: ttng.tmem_store %cst

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[128, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64], [0, 32]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 1, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 0, 1]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [64, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 88 : i32, ttg.min_reg_auto_ws = 88 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<256x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16, #shared>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<128x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<128x16xf32, #shared1>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16, #shared>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16, #shared>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<128xf32, #shared2>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<128xf32, #shared2>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32, %arg61: i32, %arg62: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.693147182> : tensor<128x16xf32, #blocked>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i64 = arith.constant 2 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = arith.muli %0, %arg62 : i32
    %3 = arith.extsi %2 : i32 to i64
    %4 = arith.remsi %0, %arg61 : i32
    %5 = arith.muli %arg58, %4 : i32
    %6 = arith.divsi %0, %arg61 : i32
    %7 = arith.muli %arg57, %6 : i32
    %8 = arith.addi %5, %7 : i32
    %9 = arith.extsi %8 : i32 to i64
    %10 = arith.extsi %arg59 : i32 to i64
    %11 = arith.divsi %9, %10 : i64
    %12 = arith.muli %1, %c128_i32 : i32
    %13 = arith.remsi %1, %c2_i32 : i32
    %14 = arith.extsi %12 : i32 to i64
    %15 = arith.addi %11, %14 : i64
    %16 = arith.trunci %15 : i64 to i32
    %17 = tt.descriptor_load %arg10[%16, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1>
    %18 = ttg.local_alloc %17 {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %19 = arith.muli %13, %c128_i32 : i32
    %20 = arith.subi %12, %19 : i32
    %21 = arith.extsi %20 : i32 to i64
    %22 = arith.addi %11, %21 : i64
    %23 = arith.trunci %22 : i64 to i32
    %24 = nvg.cluster_id
    %25 = arith.remsi %24, %c2_i32 : i32
    %26 = arith.muli %25, %c64_i32 : i32
    %27 = tt.descriptor_load %arg15[%23, %26] {ttg.partition = array<i32: 3>, two_cta_b} : !tt.tensordesc<256x64xf16, #shared> -> tensor<256x64xf16, #blocked2>
    %28 = ttg.local_alloc %27 {ttg.partition = array<i32: 3>} : (tensor<256x64xf16, #blocked2>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
    %29 = tt.descriptor_load %arg20[%16, %c0_i32] {ttg.partition = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1>
    %30 = ttg.local_alloc %29 {ttg.partition = array<i32: 3>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %31 = arith.divsi %arg62, %c128_i32 : i32
    %32 = arith.muli %13, %c64_i32 : i32
    %33 = arith.extsi %32 : i32 to i64
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_1, %token_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %34 = ttng.tmem_store %cst_0, %result_5[%token_6], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %35 = ttng.tmem_store %cst_0, %result_3[%token_4], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %nonempty = arith.cmpi sgt, %31, %c0_i32 {ttg.partition = array<i32: 1>} : i32
    llvm.intr.assume %nonempty : i1 {ttg.partition = array<i32: 1>}
    %36:7 = scf.for %arg63 = %c0_i32 to %31 step %c1_i32 iter_args(%arg64 = %c0_i32, %arg65 = %false, %arg66 = %token, %arg67 = %token_2, %arg68 = %35, %arg69 = %34, %arg70 = %token_8) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %58 = arith.extsi %arg64 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32 to i64
      %59 = arith.addi %11, %58 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i64
      %60 = arith.trunci %59 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
      %61 = arith.addi %60, %26 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %62 = tt.descriptor_load %arg5[%61, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>, two_cta_b} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %63 = ttg.local_alloc %62 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %64 = ttg.memdesc_trans %63 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared3, #smem>
      %65 = tt.descriptor_load %arg0[%60, %26] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>, two_cta_b} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked2>
      %66 = ttg.local_alloc %65 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %67 = arith.addi %3, %58 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
      %68 = arith.trunci %67 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
      %69 = tt.descriptor_load %arg51[%68] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2> -> tensor<128xf32, #blocked3>
      %70 = ttng.tc_gen5_mma %18, %64, %result[%arg66], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 0 : i32, ttg.partition = array<i32: 2>, two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %71 = ttg.convert_layout %69 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %72 = ttg.convert_layout %69 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %73 = tt.expand_dims %71 {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %74 = tt.expand_dims %72 {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %75 = tt.broadcast %73 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %result_15, %token_16 = ttng.tmem_load %result[%70] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %76 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %result_15, %75 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %77 = math.exp2 %76 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %78 = tt.descriptor_load %arg26[%60, %26] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>, two_cta_b} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked2>
      %79 = ttg.local_alloc %78 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %80 = tt.descriptor_load %arg31[%61, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>, two_cta_b} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %81 = arith.truncf %77 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %result_17 = ttng.tmem_alloc %81 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
      %82 = ttg.local_alloc %80 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %83 = ttg.memdesc_trans %82 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared3, #smem>
      %84 = ttng.tc_gen5_mma %30, %83, %result_1[%arg67], %false, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 0 : i32, ttg.partition = array<i32: 2>, two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %85 = tt.descriptor_load %arg54[%68] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2> -> tensor<128xf32, #blocked3>
      %86 = ttng.tc_gen5_mma %result_17, %79, %result_3[%arg68], %arg65, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 0 : i32, ttg.partition = array<i32: 2>, ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %87 = ttg.convert_layout %85 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %88 = ttg.convert_layout %85 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %89 = tt.expand_dims %87 {axis = 0 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %90 = tt.expand_dims %88 {axis = 0 : i32, loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %91 = tt.broadcast %89 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %result_18, %token_19 = ttng.tmem_load %result_1[%84] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %92 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 1 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %result_18, %91 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %93 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 1 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %77, %92 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %94 = arith.truncf %93 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %result_20 = ttng.tmem_alloc %94 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
      %95 = ttng.tc_gen5_mma %result_20, %66, %result_5[%arg69], %arg65, %true {loop.cluster = 1 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 0 : i32, ttg.partition = array<i32: 2>, ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %96 = ttng.two_cta_peer_gather %94 split_dim = 1 num_ctas = 2 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<64x256xf16, #linear1>
      %97 = tt.trans %96 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<64x256xf16, #linear1> -> tensor<256x64xf16, #linear2>
      %98 = tt.reshape %94 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x2x64xf16, #linear3>
      %99 = tt.trans %98 {loop.cluster = 1 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf16, #linear3> -> tensor<128x64x2xf16, #linear4>
      %100 = ttg.convert_layout %99 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf16, #linear4> -> tensor<128x64x2xf16, #blocked4>
      %outLHS_21, %outRHS_22 = tt.split %100 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf16, #blocked4> -> tensor<128x64xf16, #blocked5>
      %101 = ttg.local_alloc %outLHS_21 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked5>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %102 = ttg.local_alloc %97 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf16, #linear2>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      ttng.two_cta_peer_relay %101 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : <128x64xf16, #shared, #smem>
      %103 = ttg.memdesc_trans %102 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared3, #smem>
      %104 = ttng.tc_gen5_mma %103, %28, %result_7[%arg70], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttg.partition = array<i32: 2>, ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared3, #smem>, !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %result_23, %token_24 = ttng.tmem_load %result_7[%104] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear5>
      %105 = tt.reshape %result_23 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xf32, #linear5> -> tensor<128x2x32xf32, #linear6>
      %106 = tt.trans %105 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 1>} : tensor<128x2x32xf32, #linear6> -> tensor<128x32x2xf32, #linear7>
      %107 = ttg.convert_layout %106 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x32x2xf32, #linear7> -> tensor<128x32x2xf32, #linear8>
      %outLHS_25, %outRHS_26 = tt.split %107 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x32x2xf32, #linear8> -> tensor<128x32xf32, #linear9>
      %108 = tt.reshape %outLHS_25 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked6>
      %109 = tt.trans %108 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 1>} : tensor<128x2x16xf32, #blocked6> -> tensor<128x16x2xf32, #blocked7>
      %outLHS_27, %outRHS_28 = tt.split %109 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16x2xf32, #blocked7> -> tensor<128x16xf32, #blocked>
      %110 = tt.reshape %outRHS_26 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked6>
      %111 = tt.trans %110 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 1>} : tensor<128x2x16xf32, #blocked6> -> tensor<128x16x2xf32, #blocked7>
      %outLHS_29, %outRHS_30 = tt.split %111 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16x2xf32, #blocked7> -> tensor<128x16xf32, #blocked>
      %112 = arith.addi %59, %33 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i64
      %113 = arith.muli %112, %c2_i64 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i64
      %114 = arith.mulf %outLHS_27, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16xf32, #blocked>
      %115 = arith.trunci %113 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : i64 to i32
      %116 = ttg.local_alloc %114 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
      %117 = ttng.async_tma_reduce add, %arg36[%115, %c0_i32] %116 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %117   {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      %118 = arith.mulf %outRHS_28, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16xf32, #blocked>
      %119 = ttg.local_alloc %118 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
      %120 = ttng.async_tma_reduce add, %arg36[%115, %c16_i32] %119 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %120   {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      %121 = arith.mulf %outLHS_29, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16xf32, #blocked>
      %122 = ttg.local_alloc %121 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
      %123 = ttng.async_tma_reduce add, %arg36[%115, %c32_i32] %122 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %123   {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      %124 = arith.mulf %outRHS_30, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x16xf32, #blocked>
      %125 = ttg.local_alloc %124 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
      %126 = ttng.async_tma_reduce add, %arg36[%115, %c48_i32] %125 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %126   {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      %127 = arith.addi %arg64, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %127, %true, %token_16, %token_19, %86, %95, %token_24 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.merge_epilogue_to_computation = true, tt.scheduled_max_stage = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
    %result_9, %token_10 = ttng.tmem_load %result_3[%36#4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %result_11, %token_12 = ttng.tmem_load %result_5[%36#5] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %37 = tt.reshape %result_9 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
    %38 = tt.trans %37 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
    %39 = ttg.convert_layout %38 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #blocked8>
    %outLHS, %outRHS = tt.split %39 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #blocked8> -> tensor<128x64xf32, #blocked2>
    %40 = arith.truncf %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %41 = ttg.local_alloc %40 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %42 = ttng.async_tma_copy_local_to_global %arg46[%16, %c0_i32] %41 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %42   {ttg.partition = array<i32: 0>} : !ttg.async.token
    %43 = arith.truncf %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %44 = ttg.local_alloc %43 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %45 = ttng.async_tma_copy_local_to_global %arg46[%16, %c64_i32] %44 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %45   {ttg.partition = array<i32: 0>} : !ttg.async.token
    %46 = tt.reshape %result_11 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
    %47 = tt.trans %46 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
    %48 = ttg.convert_layout %47 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #blocked8>
    %outLHS_13, %outRHS_14 = tt.split %48 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #blocked8> -> tensor<128x64xf32, #blocked2>
    %49 = tt.splat %arg25 : f32 -> tensor<128x64xf32, #blocked2>
    %50 = arith.mulf %outLHS_13, %49 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2>
    %51 = arith.truncf %50 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %52 = ttg.local_alloc %51 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %53 = ttng.async_tma_copy_local_to_global %arg41[%16, %c0_i32] %52 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %53   {ttg.partition = array<i32: 0>} : !ttg.async.token
    %54 = arith.mulf %outRHS_14, %49 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2>
    %55 = arith.truncf %54 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %56 = ttg.local_alloc %55 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %57 = ttng.async_tma_copy_local_to_global %arg41[%16, %c64_i32] %56 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %57   {ttg.partition = array<i32: 0>} : !ttg.async.token
    tt.return
  }
}
