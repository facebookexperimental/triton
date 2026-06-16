// RUN: triton-opt --convert-triton-to-tritongpu="num-ctas=2 num-warps=4 target=cuda:90 threads-per-warp=32" --triton-nvidia-gpu-plan-cta %s | FileCheck %s

// CHECK-LABEL: @nop_kernel
// CHECK: tt.return
module {
  tt.func public @nop_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    tt.return
  }
}
