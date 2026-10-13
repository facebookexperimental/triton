
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
#layout = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 8 : i32,
  ttg.target = "cuda:90",
  "ttg.threads-per-warp" = 32 : i32
} {
  // CHECK-LABEL: tt.func private @compute_dq_tail()
  // CHECK-SAME: -> tensor<16x128xf32, #[[$LAYOUT]]>
  // CHECK: tt.return %{{.*}} : tensor<16x128xf32, #[[$LAYOUT]]>
  tt.func private @compute_dq_tail() -> tensor<16x128xf32> {
    %value = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #layout>
    %layout_marker = tlx.require_layout %value : tensor<16x128xf32, #layout> -> tensor<16x128xf32, #layout>
    tt.return %value : tensor<16x128xf32, #layout>
  }
  // CHECK-LABEL: tt.func @dq_tail_helpers(
  // CHECK: %[[DQ:.*]] = tt.call @compute_dq_tail() : () -> tensor<16x128xf32, #[[$LAYOUT]]>
  tt.func @dq_tail_helpers(%value: tensor<16x128xf32>) {
    %dq = tt.call @compute_dq_tail() : () -> tensor<16x128xf32>
    tt.return
  }
}

//--- amd.mlir

// CHECK-DAG: #[[$BUFFER_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-DAG: #[[$BUFFER_USER:.*]] = #tlx.user_layout<#[[$BUFFER_MMA]]>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
module {
  // CHECK-LABEL: tt.func private @buffer_tail_previous(
  // CHECK-SAME: -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
  tt.func private @buffer_tail_previous(%ptr: !tt.ptr<f32>, %offsets: tensor<16x128xi32>, %mask: tensor<16x128xi1>) -> tensor<16x128xf32, #pin> {
    // CHECK: %[[BUFFER_OFFSETS:.*]] = tlx.require_layout %arg1
    %pinned_offsets = tlx.require_layout %offsets : tensor<16x128xi32> -> tensor<16x128xi32, #pin>
    // CHECK: %[[BUFFER_MASK:.*]] = tlx.require_layout %arg2
    %pinned_mask = tlx.require_layout %mask : tensor<16x128xi1> -> tensor<16x128xi1, #pin>
    // CHECK: %[[BUFFER_ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #[[$BUFFER_MMA]]>
    %zero = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #mma>
    // CHECK: %[[BUFFER_OTHER:.*]] = tlx.require_layout %[[BUFFER_ZERO]] : tensor<16x128xf32, #[[$BUFFER_MMA]]> -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
    // CHECK: amdg.buffer_load %arg0[%[[BUFFER_OFFSETS]]], %[[BUFFER_MASK]], %[[BUFFER_OTHER]]
    // CHECK-SAME: contiguity = 2 : i32
    // CHECK-SAME: tlx.preserve_layout
    // CHECK-SAME: tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
    %previous = "amdg.buffer_load"(%ptr, %pinned_offsets, %pinned_mask, %zero) <{cache = 1 : i32, contiguity = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}> {tlx.preserve_layout} : (!tt.ptr<f32>, tensor<16x128xi32, #pin>, tensor<16x128xi1, #pin>, tensor<16x128xf32, #mma>) -> tensor<16x128xf32, #pin>
    // CHECK: arith.addf %[[BUFFER_ZERO]], %[[BUFFER_ZERO]] : tensor<16x128xf32, #[[$BUFFER_MMA]]>
    %bare_use = arith.addf %zero, %zero : tensor<16x128xf32, #mma>
    // CHECK: %[[BUFFER_STRIDE:.*]] = arith.constant 512 : i32
    %stride = arith.constant 512 : i32
    // CHECK: %[[BUFFER_STRIDED_OTHER:.*]] = tlx.require_layout %[[BUFFER_ZERO]] : tensor<16x128xf32, #[[$BUFFER_MMA]]> -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
    // CHECK: %[[BUFFER_STRIDED_LOAD:.*]] = amdg.buffer_load %arg0[%[[BUFFER_OFFSETS]]], %[[BUFFER_MASK]], %[[BUFFER_STRIDED_OTHER]] stride = %[[BUFFER_STRIDE]]
    // CHECK-SAME: contiguity = 2 : i32
    // CHECK-SAME: tlx.preserve_layout
    %strided_previous = "amdg.buffer_load"(%ptr, %pinned_offsets, %stride, %pinned_mask, %zero) <{cache = 1 : i32, contiguity = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> {tlx.preserve_layout} : (!tt.ptr<f32>, tensor<16x128xi32, #pin>, i32, tensor<16x128xi1, #pin>, tensor<16x128xf32, #mma>) -> tensor<16x128xf32, #pin>
    // CHECK: tt.return %[[BUFFER_STRIDED_LOAD]] : tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
    tt.return %strided_previous : tensor<16x128xf32, #pin>
  }
  // CHECK-LABEL: tt.func @buffer_tail_previous_caller(
  // CHECK: tt.call @buffer_tail_previous
  // CHECK-SAME: -> tensor<16x128xf32, #tlx.no_verify_layout<#[[$BUFFER_USER]]>>
  tt.func @buffer_tail_previous_caller(%ptr: !tt.ptr<f32>, %offsets: tensor<16x128xi32>, %mask: tensor<16x128xi1>) {
    %previous = tt.call @buffer_tail_previous(%ptr, %offsets, %mask) : (!tt.ptr<f32>, tensor<16x128xi32>, tensor<16x128xi1>) -> tensor<16x128xf32, #pin>
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
