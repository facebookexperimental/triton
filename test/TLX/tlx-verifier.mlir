
// RUN: triton-opt -split-input-file -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=8 target=cuda:90 num-ctas=2 threads-per-warp=32})' --verify-diagnostics %s

module attributes {tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @legalize_warp_partition(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    // expected-error @+1 {{WarpSpecializeOp should not capture RankedTensorType}}
    ttg.warp_specialize(%arg3, %3, %arg5)
    default {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %4 = arith.addi %3, %2 : tensor<1024xi32>
      ttg.warp_yield
    }
    partition0(%arg7: !tt.ptr<f32>, %arg8: tensor<1024xi32>, %arg9: !tt.ptr<f32>) num_warps(1) {
      %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      %4 = arith.addi %arg8, %2 : tensor<1024xi32>
      %5 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      ttg.warp_return
    } : (!tt.ptr<f32>, tensor<1024xi32>, !tt.ptr<f32>) -> ()
    tt.return
  }
}
