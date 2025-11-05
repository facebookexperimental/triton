// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=4' | FileCheck %s

// CHECK-LABEL: @legalize_warp_specialize
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @legalize_warp_specialize(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  partition0(%arg2: !tt.ptr<i32>) num_warps(2) {
    // CHECK: tt.splat {{.*}} : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
    // CHECK: tt.load {{.*}} : tensor<256x!tt.ptr<i32>, #blocked>
    %splatted = tt.splat %arg2 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %input = tt.load %splatted : tensor<256x!tt.ptr<i32>>
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  tt.return
}
}


// -----
// CHECK-DAG: [[DEFAULT:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: [[WS1:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
// CHECK: @legalize_warp_partition
module attributes {tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @legalize_warp_partition(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    ttg.warp_specialize(%arg3, %1, %arg5)
    // CHECK: default
    default {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %3 = tt.splat %1 : i32 -> tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[DEFAULT]]
      %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
      %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %9, %7 : tensor<1024x!tt.ptr<f32>>
      ttg.warp_yield
    }
    // CHECK: partition0
    partition0(%arg7: !tt.ptr<f32>, %arg8: i32, %arg9: !tt.ptr<f32>) num_warps(1) {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %3 = tt.splat %arg8 : i32 -> tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      %5 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[WS1]]
      %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
      %8 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %9, %7 : tensor<1024x!tt.ptr<f32>>
      ttg.warp_return
    } : (!tt.ptr<f32>, i32, !tt.ptr<f32>) -> ()
    tt.return
  }
}
