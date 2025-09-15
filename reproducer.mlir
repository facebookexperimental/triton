#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_warp_specialized_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>, #blocked>
    ttg.warp_specialize()
    default {
      tt.store %2, %3 : tensor<1024x!tt.ptr<f32>, #blocked>
      ttg.warp_yield
    } : () -> ()
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(convert-triton-amdgpu-to-llvm{arch=gfx942 ftz=true})",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
