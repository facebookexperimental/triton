
// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=2 threads-per-warp=32})' %s| FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: tlx.has_tlx_ops = true
// CHECK-SAME: "ttg.num-ctas" = 2
// CHECK-SAME: "ttg.num-warps" = 8
// CHECK-SAME: ttg.target = "cuda:90"
// CHECK-SAME: "ttg.threads-per-warp" = 32
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
// CHECK-SAME: tlx.has_explicit_local_mem_access = true
// CHECK-NOT: tlx.has_tlx_ops
// CHECK-SAME: "ttg.num-ctas" = 2
// CHECK-SAME: "ttg.num-warps" = 8
// CHECK-SAME: ttg.target = "cuda:90"
// CHECK-SAME: "ttg.threads-per-warp" = 32
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
    %12 = ttg.memdesc_subview %11[%c0_i32, %c0_i32] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %13 = ttg.memdesc_subview %11[%c1_i32, %c0_i32] : !ttg.memdesc<2x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
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
