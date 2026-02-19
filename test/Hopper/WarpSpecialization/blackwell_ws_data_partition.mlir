// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=3 | FileCheck %s


// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @_helion_attention_kernel
  tt.func public @_helion_attention_kernel(%q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %lse: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %lse_desc = arith.constant 8192 : i64
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %total_pids = arith.constant 4096 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.127517432> : tensor<256x128xf32, #blocked>
    %cst_2 = arith.constant dense<0.127517432> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #blocked>
    // CHECK-COUNT-8: tt.make_tensor_descriptor
    %q_desc = tt.make_tensor_descriptor %q, [%c128_i32, %c8192_i32, %c128_i32], [%c1048576_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<1x256x128xbf16, #shared>>
    %k_desc = tt.make_tensor_descriptor %k, [%c128_i32, %c8192_i32, %c128_i32], [%c1048576_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<1x128x128xbf16, #shared>>
    %v_desc = tt.make_tensor_descriptor %v, [%c128_i32, %c8192_i32, %c128_i32], [%c1048576_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<1x128x128xbf16, #shared>>
    %lse_desc_4 = tt.make_tensor_descriptor %lse, [%c128_i32, %c8192_i32], [%lse_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<tensor<1x256xf32, #shared1>>
    %o_desc = tt.make_tensor_descriptor %o, [%c128_i32, %c8192_i32, %c128_i32], [%c1048576_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<1x256x128xbf16, #shared>>
    %0 = tt.get_program_id x : i32
    scf.for %virtual_pid = %0 to %total_pids step %c148_i32  : i32 {
      %pid_0 = arith.remsi %virtual_pid, %c32_i32 : i32
      %pid_1 = arith.divsi %virtual_pid, %c32_i32 : i32
      %offset_0 = arith.muli %pid_0, %c256_i32 : i32
      %q_i_load = tt.descriptor_load %q_desc[%pid_1, %offset_0, %c0_i32] : !tt.tensordesc<tensor<1x256x128xbf16, #shared>> -> tensor<256x128xbf16, #blocked1>
      %q_i_load_5 = ttg.local_alloc %q_i_load : (tensor<256x128xbf16, #blocked1>) -> !ttg.memdesc<256x128xbf16, #shared2, #smem>
      %qk, %qk_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_8 = ttng.tmem_store %cst_3, %acc[%acc_7], %true : tensor<256x128xf32, #blocked> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_9:4 = scf.for %acc_15 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg7 = %cst_0, %arg8 = %cst, %qk_16 = %qk_6, %acc_17 = %acc_8) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_j_load = tt.descriptor_load %k_desc[%pid_1, %acc_15, %c0_i32] : !tt.tensordesc<tensor<1x128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
        %v_j_load = tt.descriptor_load %v_desc[%pid_1, %acc_15, %c0_i32] : !tt.tensordesc<tensor<1x128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
        %v_j_load_18 = ttg.local_alloc %v_j_load : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
        %permute = ttg.local_alloc %k_j_load : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
        %permute_19 = ttg.memdesc_trans %permute {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared2, #smem> -> !ttg.memdesc<128x128xbf16, #shared3, #smem>
        // CHECK-COUNT-2: ttng.tc_gen5_mma
        %qk_20 = ttng.tc_gen5_mma %q_i_load_5, %permute_19, %qk[%qk_16], %false, %true : !ttg.memdesc<256x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xbf16, #shared3, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_21, %qk_22 = ttng.tmem_load %qk[%qk_20] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked>
        %amax = "tt.reduce"(%qk_21) <{axis = 1 : i32}> ({
        ^bb0(%amax_36: f32, %amax_37: f32):
          %amax_38 = arith.maxnumf %amax_36, %amax_37 : f32
          tt.reduce.return %amax_38 : f32
        }) : (tensor<256x128xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %v_5 = arith.mulf %amax, %cst_2 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %mask = arith.cmpf ogt, %arg7, %v_5 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %mask_23 = arith.cmpf une, %arg7, %arg7 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %mask_24 = arith.ori %mask, %mask_23 : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
        %v_6 = arith.select %mask_24, %arg7, %v_5 : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %v_8 = arith.mulf %qk_21, %cst_1 : tensor<256x128xf32, #blocked>
        %subscript = tt.expand_dims %v_6 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
        %v_9 = tt.broadcast %subscript : tensor<256x1xf32, #blocked> -> tensor<256x128xf32, #blocked>
        %v_9_25 = arith.subf %v_8, %v_9 : tensor<256x128xf32, #blocked>
        %v_10 = tt.extern_elementwise %v_9_25 {libname = "", libpath = "", pure = true, symbol = "__nv_exp2f"} : (tensor<256x128xf32, #blocked>) -> tensor<256x128xf32, #blocked>
        %v_11 = arith.subf %arg7, %v_6 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %v_12 = tt.extern_elementwise %v_11 {libname = "", libpath = "", pure = true, symbol = "__nv_exp2f"} : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij = "tt.reduce"(%v_10) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_36: f32, %l_ij_37: f32):
          %l_ij_38 = arith.addf %l_ij_36, %l_ij_37 : f32
          tt.reduce.return %l_ij_38 : f32
        }) : (tensor<256x128xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_26, %acc_27 = ttng.tmem_load %acc[%acc_17] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked>
        %1 = tt.reshape %acc_26 : tensor<256x128xf32, #blocked> -> tensor<256x2x64xf32, #blocked2>
        %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<256x2x64xf32, #blocked2> -> tensor<256x64x2xf32, #blocked3>
        %outLHS, %outRHS = tt.split %2 : tensor<256x64x2xf32, #blocked3> -> tensor<256x64xf32, #blocked4>
        %acc0 = tt.expand_dims %v_12 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
        %acc0_28 = ttg.convert_layout %acc0 : tensor<256x1xf32, #blocked> -> tensor<256x1xf32, #blocked4>
        %acc0_29 = tt.broadcast %acc0_28 : tensor<256x1xf32, #blocked4> -> tensor<256x64xf32, #blocked4>
        %acc0_30 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %outLHS, %acc0_29 : tensor<256x64xf32, #blocked4>, tensor<256x64xf32, #blocked4> -> tensor<256x64xf32, #blocked4>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %outRHS, %acc0_29 : tensor<256x64xf32, #blocked4>, tensor<256x64xf32, #blocked4> -> tensor<256x64xf32, #blocked4>
                %inline_triton_result_3 = tt.join %acc0_30, %acc1 : tensor<256x64xf32, #blocked4> -> tensor<256x64x2xf32, #blocked3>
        %inline_triton_result_3_31 = tt.trans %inline_triton_result_3 {order = array<i32: 0, 2, 1>} : tensor<256x64x2xf32, #blocked3> -> tensor<256x2x64xf32, #blocked2>
        %inline_triton_result_3_32 = tt.reshape %inline_triton_result_3_31 : tensor<256x2x64xf32, #blocked2> -> tensor<256x128xf32, #blocked>
        %v_13 = arith.truncf %v_10 : tensor<256x128xf32, #blocked> to tensor<256x128xbf16, #blocked>
        %acc_33 = ttng.tmem_alloc %v_13 : (tensor<256x128xbf16, #blocked>) -> !ttg.memdesc<256x128xbf16, #tmem1, #ttng.tensor_memory>
        %acc_34 = ttng.tmem_store %inline_triton_result_3_32, %acc[%acc_27], %true : tensor<256x128xf32, #blocked> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK-COUNT-2: ttng.tc_gen5_mma
        %acc_35 = ttng.tc_gen5_mma %acc_33, %v_j_load_18, %acc[%acc_34], %true, %true : !ttg.memdesc<256x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %v_14 = arith.mulf %arg8, %v_12 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %v_3 = arith.addf %v_14, %l_ij : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %v_6, %v_3, %qk_22, %acc_35 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      } {tt.disallow_acc_multi_buffer}
      %v_16 = tt.extern_elementwise %acc_9#1 {libname = "", libpath = "", pure = true, symbol = "__nv_log2f"} : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %v_17 = arith.addf %acc_9#0, %v_16 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %subscript_1 = tt.expand_dims %acc_9#1 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %v_18 = tt.broadcast %subscript_1 : tensor<256x1xf32, #blocked> -> tensor<256x128xf32, #blocked>
      %acc_10, %acc_11 = ttng.tmem_load %acc[%acc_9#3] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked>
      %v_18_12 = arith.divf %acc_10, %v_18 : tensor<256x128xf32, #blocked>
      %subscript_2 = ttg.convert_layout %v_17 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %subscript_2_13 = tt.expand_dims %subscript_2 {axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xf32, #blocked1>
      // CHECK-COUNT-2: tt.descriptor_store
      tt.descriptor_store %lse_desc_4[%pid_1, %offset_0], %subscript_2_13 : !tt.tensordesc<tensor<1x256xf32, #shared1>>, tensor<1x256xf32, #blocked1>
      %subscript_3 = ttg.convert_layout %v_18_12 : tensor<256x128xf32, #blocked> -> tensor<256x128xf32, #ttg.slice<{dim = 0, parent = #blocked5}>>
      %subscript_3_14 = tt.expand_dims %subscript_3 {axis = 0 : i32} : tensor<256x128xf32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x256x128xf32, #blocked5>
      %v_19 = arith.truncf %subscript_3_14 : tensor<1x256x128xf32, #blocked5> to tensor<1x256x128xbf16, #blocked5>
      // CHECK-COUNT-2: tt.descriptor_store
      tt.descriptor_store %o_desc[%pid_1, %offset_0, %c0_i32], %v_19 : !tt.tensordesc<tensor<1x256x128xbf16, #shared>>, tensor<1x256x128xbf16, #blocked5>
    } {tt.warp_specialize}
    tt.return
  }
}
