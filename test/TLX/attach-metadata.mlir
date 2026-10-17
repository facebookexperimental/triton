
// RUN: split-file %s %t
// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=1 threads-per-warp=32})' %t/cuda.mlir | FileCheck %t/cuda.mlir
// RUN: triton-opt -split-input-file --mlir-very-unsafe-disable-verifier-on-parsing --verify-each -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=1 threads-per-warp=32})' %t/cuda-helper-abi.mlir | FileCheck %t/cuda-helper-abi.mlir
// RUN: triton-opt -split-input-file --mlir-very-unsafe-disable-verifier-on-parsing --verify-each -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=hip:gfx950 num-ctas=1 threads-per-warp=64})' %t/amd.mlir | FileCheck %t/amd.mlir
// RUN: not triton-opt --mlir-very-unsafe-disable-verifier-on-parsing --verify-each -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=hip:gfx950 num-ctas=1 threads-per-warp=64})' %t/amd-buffer-conflict.mlir 2>&1 | FileCheck %t/amd-buffer-conflict.mlir

//--- cuda.mlir

// CHECK: module attributes {
// CHECK-DAG: triton.skip_generic_pipeline
// CHECK-DAG: tlx.has_tlx_ops = true
// CHECK-DAG: "ttg.num-ctas" = 1
// CHECK-DAG: "ttg.num-warps" = 8
// CHECK-DAG: ttg.target = "cuda:90"
// CHECK-DAG: "ttg.threads-per-warp" = 32
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
module {
    tt.func @kernel_tlx(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
    %1 = tt.splat %cst : f32 -> tensor<256xf32>
    %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
        %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
        %4 = arith.addf %arg4, %3 : tensor<256xf32>
        %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
        scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
    } {tt.loop_unroll_factor = 2 : i32}
    // manually inserted tlx.require_layout here. This TTIR is not necessarily a valid kernel
    %51 = "tlx.require_layout"(%0) : (tensor<256xi32>) -> tensor<256xi32, #blocked>
    tt.return
    }
}

// -----

// CHECK: module {
// CHECK-NOT: triton.skip_generic_pipeline
// CHECK-NOT: tlx.has_explicit_local_mem_access
// CHECK-NOT: tlx.has_tlx_ops
// CHECK-NOT: "ttg.num-ctas"
// CHECK-NOT: "ttg.num-warps"
module {
    tt.func @kernel_no_tlx(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
    %1 = tt.splat %cst : f32 -> tensor<256xf32>
    %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
        %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
        %4 = arith.addf %arg4, %3 : tensor<256xf32>
        %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
        scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
    } {tt.loop_unroll_factor = 2 : i32}
    tt.return
    }
}

// -----

// CHECK: module attributes {
// CHECK-DAG: triton.skip_generic_pipeline
// CHECK-DAG: tlx.has_explicit_local_mem_access = true
// CHECK-DAG: "ttg.num-ctas" = 1
// CHECK-DAG: "ttg.num-warps" = 8
// CHECK-DAG: ttg.target = "cuda:90"
// CHECK-DAG: "ttg.threads-per-warp" = 32
// CHECK-NOT: tlx.has_tlx_ops
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @local_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg3: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %1 : i32 -> tensor<64xi32>
    %4 = arith.addi %3, %2 : tensor<64xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %12 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %13 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %14 = ttg.async_copy_global_to_local %8, %12 mask %6 : tensor<64x!tt.ptr<f32>> -> <64xf32, #shared, #smem, mutable>
    %15 = ttg.async_copy_global_to_local %10, %13 mask %6 : tensor<64x!tt.ptr<f32>> -> <64xf32, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_wait  {num = 0 : i32}
    %18 = ttg.local_load %12 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
    %19 = ttg.local_load %13 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
    %20 = arith.addf %18, %19 : tensor<64xf32>
    %21 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %22 = tt.addptr %21, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %22, %20, %6 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A suspend hint on tlx.async_tasks is propagated from warp_specialize to the
// module so waits introduced by later warp-specialization passes also see it.
// CHECK: module attributes {
// CHECK-SAME: tlx.mbarrier_try_wait_suspend_ns = 50000 : i32
module {
  tt.func @kernel_mbarrier_try_wait_suspend() {
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 50000 : i32}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}


// -----

// CHECK: module attributes {
// CHECK-DAG: triton.skip_generic_pipeline
// CHECK-DAG: tlx.has_warp_spec_ops = true
// CHECK-NOT: tlx.has_explicit_local_mem_access
// CHECK-NOT: tlx.has_tlx_ops
module attributes {tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add2_warp_specialized_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg6: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    ttg.warp_specialize(%arg3, %arg4, %1, %arg5, %arg6) attributes {requestedRegisters = array<i32: 100, 100>}
    default {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %3 = tt.splat %1 : i32 -> tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      %5 = tt.splat %arg6 : i32 -> tensor<1024xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
      %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
      %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
      %13 = arith.addf %9, %12 : tensor<1024xf32>
      %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
      ttg.warp_yield
    }
    partition0(%arg7: !tt.ptr<f32> , %arg8: !tt.ptr<f32> , %arg9: i32 , %arg10: !tt.ptr<f32> , %arg11: i32 ) num_warps(4) {
      %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %3 = tt.splat %arg9 : i32 -> tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      %5 = tt.splat %arg11 : i32 -> tensor<1024xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
      %7 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
      %10 = tt.splat %arg8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
      %13 = arith.addf %9, %cst : tensor<1024xf32>
      %14 = arith.addf %13, %12 : tensor<1024xf32>
      %15 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %16, %14, %6 : tensor<1024x!tt.ptr<f32>>
      ttg.warp_return
    }
    partition1(%arg7: !tt.ptr<f32> , %arg8: !tt.ptr<f32> , %arg9: i32 , %arg10: !tt.ptr<f32> , %arg11: i32 ) num_warps(4) {
      %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32>
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %3 = tt.splat %arg9 : i32 -> tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      %5 = tt.splat %arg11 : i32 -> tensor<1024xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
      %7 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
      %10 = tt.splat %arg8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
      %13 = arith.addf %9, %cst : tensor<1024xf32>
      %14 = arith.subf %12, %cst : tensor<1024xf32>
      %15 = arith.addf %13, %14 : tensor<1024xf32>
      %16 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %17 = tt.addptr %16, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %17, %15, %6 : tensor<1024x!tt.ptr<f32>>
      ttg.warp_return
    } : (!tt.ptr<f32>, !tt.ptr<f32>, i32, !tt.ptr<f32>, i32) -> ()
    tt.return
  }
}

// -----

// CHECK: module attributes {
// CHECK-SAME: tlx.enable_paired_cta_mma = true
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, ctaMode = twocta_rhs>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttng.two-ctas" = true} {
  tt.func @tc_gen5_mma(%a: !ttg.memdesc<256x128xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async, two_ctas}:
       !ttg.memdesc<256x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

// A `tlx.no_ending_cluster_sync` marker on the warp_specialize op is propagated
// to the `tlx.user_post_ws_sync` module attribute (NVIDIA target).
// CHECK: module attributes {
// CHECK-SAME: tlx.user_post_ws_sync = true
module {
  tt.func @kernel_no_ending_cluster_sync() {
    ttg.warp_specialize() attributes {tlx.no_ending_cluster_sync}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
//--- cuda-helper-abi.mlir

// Frontend helper signatures have no tensor encodings.
// CHECK-DAG: #[[$LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-DAG: #[[$PIN:.*]] = #tlx.user_layout<#[[$LAYOUT]]>
#layout = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#layout>>
module {
  // CHECK-LABEL: tt.func private @compute_dq_tail()
  // CHECK-SAME: -> tensor<16x128xf32, #[[$LAYOUT]]>
  // CHECK: tt.return %{{.*}} : tensor<16x128xf32, #[[$LAYOUT]]>
  tt.func private @compute_dq_tail() -> tensor<16x128xf32> {
    %value = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #layout>
    tt.return %value : tensor<16x128xf32, #layout>
  }
  // CHECK-LABEL: tt.func private @load_previous(
  // CHECK-SAME: -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>
  // CHECK: %[[PINNED:.*]] = tlx.require_layout
  // CHECK: tt.return %[[PINNED]] : tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>
  tt.func private @load_previous(%value: tensor<16x128xf32>) -> tensor<16x128xf32> {
    %pinned = tlx.require_layout %value : tensor<16x128xf32> -> tensor<16x128xf32, #pin>
    tt.return %pinned : tensor<16x128xf32, #pin>
  }
  // CHECK-LABEL: tt.func private @store_final_preloaded(
  // CHECK-SAME: %{{[^:]+}}: tensor<16x128xf32, #[[$LAYOUT]]>, %{{[^:]+}}: tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>)
  // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>
  tt.func private @store_final_preloaded(%dq: tensor<16x128xf32>, %previous: tensor<16x128xf32>) {
    %required = tlx.require_layout %dq : tensor<16x128xf32> -> tensor<16x128xf32, #layout>
    %partial = tlx.require_layout %required : tensor<16x128xf32, #layout> -> tensor<16x128xf32, #pin>
    %combined = "arith.addf"(%previous, %partial) : (tensor<16x128xf32>, tensor<16x128xf32, #pin>) -> tensor<16x128xf32>
    tt.return
  }
  // CHECK-LABEL: tt.func @dq_tail_helpers(
  // CHECK: %[[DQ:.*]] = tt.call @compute_dq_tail() : () -> tensor<16x128xf32, #[[$LAYOUT]]>
  // CHECK: %[[PREVIOUS:.*]] = tt.call @load_previous
  // CHECK-SAME: -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>
  // CHECK: tt.call @store_final_preloaded(%[[DQ]], %[[PREVIOUS]]) : (tensor<16x128xf32, #[[$LAYOUT]]>, tensor<16x128xf32, #tlx.no_verify_layout<#[[$PIN]]>>) -> ()
  tt.func @dq_tail_helpers(%value: tensor<16x128xf32>) {
    %dq = tt.call @compute_dq_tail() : () -> tensor<16x128xf32>
    %previous = tt.call @load_previous(%value) : (tensor<16x128xf32>) -> tensor<16x128xf32>
    tt.call @store_final_preloaded(%dq, %previous) : (tensor<16x128xf32>, tensor<16x128xf32>) -> ()
    tt.return
  }
}

// -----

// An explicit concrete scale constrains the product and cast layouts.
// CHECK-DAG: #[[$SCALE_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
#layout = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module {
  // CHECK-LABEL: tt.func private @scale_dkdv_tail(
  // CHECK-SAME: %{{[^:]+}}: tensor<16x128xf32, #[[$SCALE_LAYOUT]]>, %[[DV_ARG:[^:]+]]: tensor<16x128xf32, #[[$SCALE_LAYOUT]]>) -> tensor<16x128xbf16, #[[$SCALE_LAYOUT]]>
  // CHECK: %[[PRODUCT:.*]] = arith.mulf %{{.*}}, %{{.*}} : tensor<16x128xf32, #[[$SCALE_LAYOUT]]>
  // CHECK: arith.truncf %[[PRODUCT]] : tensor<16x128xf32, #[[$SCALE_LAYOUT]]> to tensor<16x128xbf16, #[[$SCALE_LAYOUT]]>
  // CHECK: arith.truncf %[[DV_ARG]] : tensor<16x128xf32, #[[$SCALE_LAYOUT]]> to tensor<16x128xbf16, #[[$SCALE_LAYOUT]]>
  // CHECK: tt.return %{{.*}} : tensor<16x128xbf16, #[[$SCALE_LAYOUT]]>
  tt.func private @scale_dkdv_tail(%dk: tensor<16x128xf32>, %dv: tensor<16x128xf32>) -> tensor<16x128xbf16> {
    %scale = arith.constant dense<0.500000e+00> : tensor<16x128xf32>
    %required = tlx.require_layout %scale : tensor<16x128xf32> -> tensor<16x128xf32, #layout>
    %scaled = "arith.mulf"(%dk, %required) : (tensor<16x128xf32>, tensor<16x128xf32, #layout>) -> tensor<16x128xf32>
    %bf16 = "arith.truncf"(%scaled) : (tensor<16x128xf32>) -> tensor<16x128xbf16>
    %output = tlx.require_layout %bf16 : tensor<16x128xbf16> -> tensor<16x128xbf16, #layout>
    %dv_bf16 = "arith.truncf"(%dv) : (tensor<16x128xf32>) -> tensor<16x128xbf16>
    %dv_output = tlx.require_layout %dv_bf16 : tensor<16x128xbf16> -> tensor<16x128xbf16, #layout>
    tt.return %output : tensor<16x128xbf16, #layout>
  }
  // CHECK-LABEL: tt.func @scale_dkdv_caller(
  // CHECK: tt.call @scale_dkdv_tail
  // CHECK-SAME: : (tensor<16x128xf32, #[[$SCALE_LAYOUT]]>, tensor<16x128xf32, #[[$SCALE_LAYOUT]]>) -> tensor<16x128xbf16, #[[$SCALE_LAYOUT]]>
  tt.func @scale_dkdv_caller(%dk: tensor<16x128xf32, #layout>, %dv: tensor<16x128xf32, #layout>) {
    %scaled = tt.call @scale_dkdv_tail(%dk, %dv) : (tensor<16x128xf32, #layout>, tensor<16x128xf32, #layout>) -> tensor<16x128xbf16>
    tt.return
  }
}

//--- amd.mlir

// CHECK-DAG: #{{.*}} = #tlx.user_layout<#{{.*}}>
#mixed_dot_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#mixed_dot_a = #ttg.dot_op<{opIdx = 0, parent = #mixed_dot_mma, kWidth = 8}>
#mixed_dot_b = #ttg.dot_op<{opIdx = 1, parent = #mixed_dot_mma, kWidth = 8}>
#mixed_dot_a_pin = #tlx.no_verify_layout<#tlx.user_layout<#mixed_dot_a>>
module {
  // CHECK-LABEL: tt.func @mixed_no_verify_dot(
  tt.func @mixed_no_verify_dot(
      %a: tensor<16x128xbf16, #mixed_dot_a>,
      %b: tensor<128x16xbf16, #mixed_dot_b>,
      %c: tensor<16x16xf32, #mixed_dot_mma>) {
    // CHECK: %[[PINNED_A:.*]] = tlx.require_layout
    // CHECK-SAME: -> tensor<16x128xbf16, #tlx.no_verify_layout<
    %pinned_a = tlx.require_layout %a : tensor<16x128xbf16, #mixed_dot_a> -> tensor<16x128xbf16, #mixed_dot_a_pin>
    // CHECK: tt.dot %[[PINNED_A]], %arg1, %arg2
    %dot = tt.dot %pinned_a, %b, %c : tensor<16x128xbf16, #mixed_dot_a_pin> * tensor<128x16xbf16, #mixed_dot_b> -> tensor<16x16xf32, #mixed_dot_mma>
    tt.return
  }
  // CHECK-LABEL: tt.func @unencoded_dot_with_tlx(
  tt.func @unencoded_dot_with_tlx(%a: tensor<16x128xbf16>, %b: tensor<128x16xbf16>, %c: tensor<16x16xf32>) {
    %required = tlx.require_layout %a : tensor<16x128xbf16> -> tensor<16x128xbf16, #mixed_dot_a>
    // CHECK: tt.dot %arg0, %arg1, %arg2
    // CHECK-SAME: -> tensor<16x16xf32>
    %dot = tt.dot %a, %b, %c : tensor<16x128xbf16> * tensor<128x16xbf16> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$DECODER_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-DAG: #[[$DECODER_TRANSPOSE:.*]] = #ttg.linear<
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module {
  // CHECK-LABEL: tt.func @mfma_dot_scale_transpose(
  // CHECK-SAME: %[[DECODER_MASK:[^:]+]]: tensor<16x16xi1>)
  tt.func @mfma_dot_scale_transpose(%a: tensor<16x128xbf16, #a>, %b: tensor<128x16xbf16, #b>, %acc: tensor<16x16xf32, #mma>, %mask: tensor<16x16xi1>) {
    %required_acc = tlx.require_layout %acc : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    // CHECK: %[[QK:.*]] = tt.dot
    // CHECK-SAME: -> tensor<16x16xf32, #[[$DECODER_MMA]]>
    %qk = tt.dot %a, %b, %acc : tensor<16x128xbf16, #a> * tensor<128x16xbf16, #b> -> tensor<16x16xf32, #mma>
    %scale = arith.constant dense<0.500000e+00> : tensor<16x16xf32>
    // CHECK: %[[SCALE_EDGE:.*]] = tlx.require_layout
    // CHECK-SAME: tensor<16x16xf32> -> tensor<16x16xf32, #[[$DECODER_MMA]]>
    // CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[SCALE_EDGE]]
    // CHECK-SAME: tensor<16x16xf32, #[[$DECODER_MMA]]>
    %scaled = "arith.mulf"(%qk, %scale) : (tensor<16x16xf32, #mma>, tensor<16x16xf32>) -> tensor<16x16xf32, #mma>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    // CHECK: %[[MASK_EDGE:.*]] = tlx.require_layout %[[DECODER_MASK]]
    // CHECK-SAME: tensor<16x16xi1> -> tensor<16x16xi1, #[[$DECODER_MMA]]>
    // CHECK: %[[SELECTED:.*]] = arith.select %[[MASK_EDGE]], %[[SCALED]], %{{[^ ,:]+}}
    // CHECK-SAME: tensor<16x16xf32, #[[$DECODER_MMA]]>
    %selected = "arith.select"(%mask, %scaled, %zero) : (tensor<16x16xi1>, tensor<16x16xf32, #mma>, tensor<16x16xf32>) -> tensor<16x16xf32, #mma>
    // CHECK: %[[BF16:.*]] = arith.truncf %[[SELECTED]]
    // CHECK-SAME: tensor<16x16xf32, #[[$DECODER_MMA]]> to tensor<16x16xbf16, #[[$DECODER_MMA]]>
    %bf16 = "arith.truncf"(%selected) : (tensor<16x16xf32, #mma>) -> tensor<16x16xbf16>
    // CHECK: tt.trans %[[BF16]]
    // CHECK-SAME: -> tensor<16x16xbf16, #[[$DECODER_TRANSPOSE]]>
    %transposed = "tt.trans"(%bf16) <{order = array<i32: 1, 0>}> : (tensor<16x16xbf16>) -> tensor<16x16xbf16>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$LOOP_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module {
  // CHECK-LABEL: tt.func @mfma_loop_accumulator_scale(
  // CHECK-SAME: %[[LOOP_ACC:[^:]+]]: tensor<128x128xf32, #[[$LOOP_MMA]]>, %[[LOOP_ALPHA:[^:]+]]: tensor<128x128xf32>,
  tt.func @mfma_loop_accumulator_scale(%a: tensor<128x64xbf16, #a>, %b: tensor<64x128xbf16, #b>, %acc: tensor<128x128xf32, #mma>, %alpha: tensor<128x128xf32>, %upper: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: %[[INIT_ALPHA:.*]] = tlx.require_layout %[[LOOP_ALPHA]]
    // CHECK-SAME: tensor<128x128xf32> -> tensor<128x128xf32, #[[$LOOP_MMA]]>
    // CHECK: %[[INITIAL:.*]] = arith.mulf %[[LOOP_ACC]], %[[INIT_ALPHA]] : tensor<128x128xf32, #[[$LOOP_MMA]]>
    %initial = "arith.mulf"(%acc, %alpha) : (tensor<128x128xf32, #mma>, tensor<128x128xf32>) -> tensor<128x128xf32, #mma>
    // CHECK: %[[LOOP_RESULT:.*]] = scf.for %{{[^ ]+}} = %{{.*}} iter_args(%[[LOOP_ITER:[^ ]+]] = %[[INITIAL]]) -> (tensor<128x128xf32, #[[$LOOP_MMA]]>)
    %result = scf.for %iv = %c0 to %upper step %c1 iter_args(%iter = %initial) -> (tensor<128x128xf32, #mma>) : i32 {
      // CHECK: %[[ITER_ALPHA:.*]] = tlx.require_layout %[[LOOP_ALPHA]]
      // CHECK-SAME: tensor<128x128xf32> -> tensor<128x128xf32, #[[$LOOP_MMA]]>
      // CHECK: %[[ITER_SCALED:.*]] = arith.mulf %[[LOOP_ITER]], %[[ITER_ALPHA]] : tensor<128x128xf32, #[[$LOOP_MMA]]>
      %scaled = "arith.mulf"(%iter, %alpha) : (tensor<128x128xf32, #mma>, tensor<128x128xf32>) -> tensor<128x128xf32, #mma>
      // CHECK: %[[LOOP_DOT:.*]] = tt.dot %arg0, %arg1, %[[ITER_SCALED]]
      %dot = tt.dot %a, %b, %scaled : tensor<128x64xbf16, #a> * tensor<64x128xbf16, #b> -> tensor<128x128xf32, #mma>
      // CHECK: %[[YIELD_ALPHA:.*]] = tlx.require_layout %[[LOOP_ALPHA]]
      // CHECK-SAME: tensor<128x128xf32> -> tensor<128x128xf32, #[[$LOOP_MMA]]>
      // CHECK: %[[YIELDED:.*]] = arith.mulf %[[LOOP_DOT]], %[[YIELD_ALPHA]] : tensor<128x128xf32, #[[$LOOP_MMA]]>
      %yielded = "arith.mulf"(%dot, %alpha) : (tensor<128x128xf32, #mma>, tensor<128x128xf32>) -> tensor<128x128xf32, #mma>
      // CHECK: scf.yield %[[YIELDED]] : tensor<128x128xf32, #[[$LOOP_MMA]]>
      scf.yield %yielded : tensor<128x128xf32, #mma>
    }
    // CHECK: %[[RESULT_ALPHA:.*]] = tlx.require_layout %[[LOOP_ALPHA]]
    // CHECK-SAME: tensor<128x128xf32> -> tensor<128x128xf32, #[[$LOOP_MMA]]>
    // CHECK: %[[RESULT_SCALED:.*]] = arith.mulf %[[LOOP_RESULT]], %[[RESULT_ALPHA]] : tensor<128x128xf32, #[[$LOOP_MMA]]>
    %result_scaled = "arith.mulf"(%result, %alpha) : (tensor<128x128xf32, #mma>, tensor<128x128xf32>) -> tensor<128x128xf32, #mma>
    %released = tlx.release_layout %result_scaled : tensor<128x128xf32, #mma> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$STORE_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
module {
  // CHECK-LABEL: tt.func @mfma_store_operand_bridges(
  // CHECK-SAME: %[[STORE_PTR:[^:]+]]: tensor<16x128x!tt.ptr<bf16>>, %[[STORE_MASK:[^:]+]]: tensor<16x128xi1>,
  tt.func @mfma_store_operand_bridges(%ptr: tensor<16x128x!tt.ptr<bf16>>, %mask: tensor<16x128xi1>, %value: tensor<16x128xf32, #mma>) {
    %bf16 = arith.truncf %value : tensor<16x128xf32, #mma> to tensor<16x128xbf16, #mma>
    // CHECK: %[[STORE_VALUE:.*]] = tlx.require_layout
    %required = tlx.require_layout %bf16 : tensor<16x128xbf16, #mma> -> tensor<16x128xbf16, #mma>
    // CHECK: %[[STORE_PTR_EDGE:.*]] = tlx.require_layout %[[STORE_PTR]]
    // CHECK-SAME: tensor<16x128x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>, #[[$STORE_MMA]]>
    // CHECK: %[[STORE_MASK_EDGE:.*]] = tlx.require_layout %[[STORE_MASK]]
    // CHECK-SAME: tensor<16x128xi1> -> tensor<16x128xi1, #[[$STORE_MMA]]>
    // CHECK: tt.store %[[STORE_PTR_EDGE]], %[[STORE_VALUE]], %[[STORE_MASK_EDGE]] : tensor<16x128x!tt.ptr<bf16>, #[[$STORE_MMA]]>
    "tt.store"(%ptr, %required, %mask) <{cache = 1 : i32, evict = 1 : i32}> : (tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xbf16, #mma>, tensor<16x128xi1>) -> ()
    // CHECK: %[[STORE_RELEASED:.*]] = tlx.release_layout %[[STORE_VALUE]]
    %released = tlx.release_layout %required : tensor<16x128xbf16, #mma> -> tensor<16x128xbf16>
    // CHECK: tt.store %[[STORE_PTR]], %[[STORE_RELEASED]], %[[STORE_MASK]] : tensor<16x128x!tt.ptr<bf16>>
    tt.store %ptr, %released, %mask : tensor<16x128x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$BUFFER_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-DAG: #[[$BUFFER_USER:.*]] = #tlx.user_layout<#[[$BUFFER_MMA]]>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
module {
  // CHECK-LABEL: tt.func private @buffer_tail_previous(
  // CHECK-SAME: -> tensor<16x128xf32, #[[$BUFFER_MMA]]>
  tt.func private @buffer_tail_previous(%ptr: !tt.ptr<f32>, %offsets: tensor<16x128xi32>, %mask: tensor<16x128xi1>) -> tensor<16x128xf32> {
    // CHECK: %[[BUFFER_OFFSETS:.*]] = tlx.require_layout %arg1
    %pinned_offsets = tlx.require_layout %offsets : tensor<16x128xi32> -> tensor<16x128xi32, #pin>
    // CHECK: %[[BUFFER_MASK:.*]] = tlx.require_layout %arg2
    %pinned_mask = tlx.require_layout %mask : tensor<16x128xi1> -> tensor<16x128xi1, #pin>
    // CHECK: %[[BUFFER_ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #[[$BUFFER_MMA]]>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #mma>
    // CHECK: %[[BUFFER_PHYSICAL_OFFSETS:.*]] = tlx.require_layout %[[BUFFER_OFFSETS]] : tensor<16x128xi32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>> -> tensor<16x128xi32, #[[$BUFFER_MMA]]>
    // CHECK: %[[BUFFER_PHYSICAL_MASK:.*]] = tlx.require_layout %[[BUFFER_MASK]] : tensor<16x128xi1, #tlx.no_verify_layout<#[[$BUFFER_USER]]>> -> tensor<16x128xi1, #[[$BUFFER_MMA]]>
    // CHECK: amdg.buffer_load %arg0[%[[BUFFER_PHYSICAL_OFFSETS]]], %[[BUFFER_PHYSICAL_MASK]], %[[BUFFER_ZERO]]
    // CHECK-SAME: contiguity = 2 : i32
    // CHECK-SAME: tlx.preserve_layout
    // CHECK-SAME: tensor<16x128xf32, #[[$BUFFER_MMA]]>
    %previous = "amdg.buffer_load"(%ptr, %pinned_offsets, %pinned_mask, %zero) <{cache = 1 : i32, contiguity = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}> {tlx.preserve_layout} : (!tt.ptr<f32>, tensor<16x128xi32, #pin>, tensor<16x128xi1, #pin>, tensor<16x128xf32, #mma>) -> tensor<16x128xf32, #pin>
    // CHECK: arith.addf %[[BUFFER_ZERO]], %[[BUFFER_ZERO]] : tensor<16x128xf32, #[[$BUFFER_MMA]]>
    %bare_use = arith.addf %zero, %zero : tensor<16x128xf32, #mma>
    // CHECK: %[[BUFFER_STRIDE:.*]] = arith.constant 512 : i32
    %stride = arith.constant 512 : i32
    // CHECK: %[[BUFFER_STRIDED_OFFSETS:.*]] = tlx.require_layout %[[BUFFER_OFFSETS]] : tensor<16x128xi32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>> -> tensor<16x128xi32, #[[$BUFFER_MMA]]>
    // CHECK: %[[BUFFER_STRIDED_MASK:.*]] = tlx.require_layout %[[BUFFER_MASK]] : tensor<16x128xi1, #tlx.no_verify_layout<#[[$BUFFER_USER]]>> -> tensor<16x128xi1, #[[$BUFFER_MMA]]>
    // CHECK: %[[BUFFER_STRIDED_LOAD:.*]] = amdg.buffer_load %arg0[%[[BUFFER_STRIDED_OFFSETS]]], %[[BUFFER_STRIDED_MASK]], %[[BUFFER_ZERO]] stride = %[[BUFFER_STRIDE]]
    // CHECK-SAME: contiguity = 2 : i32
    // CHECK-SAME: tlx.preserve_layout
    %strided_previous = "amdg.buffer_load"(%ptr, %pinned_offsets, %stride, %pinned_mask, %zero) <{cache = 1 : i32, contiguity = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> {tlx.preserve_layout} : (!tt.ptr<f32>, tensor<16x128xi32, #pin>, i32, tensor<16x128xi1, #pin>, tensor<16x128xf32, #mma>) -> tensor<16x128xf32, #pin>
    // CHECK: tt.return %[[BUFFER_STRIDED_LOAD]] : tensor<16x128xf32, #[[$BUFFER_MMA]]>
    tt.return %strided_previous : tensor<16x128xf32, #pin>
  }
  // CHECK-LABEL: tt.func @buffer_tail_previous_caller(
  // CHECK: tt.call @buffer_tail_previous
  // CHECK-SAME: -> tensor<16x128xf32, #[[$BUFFER_MMA]]>
  tt.func @buffer_tail_previous_caller(%ptr: !tt.ptr<f32>, %offsets: tensor<16x128xi32>, %mask: tensor<16x128xi1>) {
    %previous = tt.call @buffer_tail_previous(%ptr, %offsets, %mask) : (!tt.ptr<f32>, tensor<16x128xi32>, tensor<16x128xi1>) -> tensor<16x128xf32>
    tt.return
  }
}

//--- amd-buffer-conflict.mlir

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#other_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = false}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#other_pin = #tlx.no_verify_layout<#tlx.user_layout<#other_mma>>
module {
  tt.func @buffer_load_pinned_fallback_conflict(%ptr: !tt.ptr<f32>, %offsets: tensor<16x128xi32>, %mask: tensor<16x128xi1, #pin>, %other: tensor<16x128xf32, #other_pin>) {
    %pinned_offsets = tlx.require_layout %offsets : tensor<16x128xi32> -> tensor<16x128xi32, #pin>
    // CHECK: error: conflicting user-pinned layouts meet on this operation
    %value = "amdg.buffer_load"(%ptr, %pinned_offsets, %mask, %other) <{cache = 1 : i32, contiguity = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}> {tlx.preserve_layout} : (!tt.ptr<f32>, tensor<16x128xi32, #pin>, tensor<16x128xi1, #pin>, tensor<16x128xf32, #other_pin>) -> tensor<16x128xf32, #pin>
    tt.return
  }
}
