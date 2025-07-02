
// RUN: triton-opt -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=2 threads-per-warp=32})' %s| FileCheck %s

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
