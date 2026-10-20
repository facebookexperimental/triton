// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory=arch=gfx950 -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-warp-pipeline=gfx-arch=gfx950 -verify-diagnostics

// The NPOT reject runs in both AMD passes: allocation, and
// convert-warp-pipeline, which builds its own ModuleAllocation.

#b = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k1_2x320_axis1(%x: tensor<2x320xf32, #b>) {
    // expected-error @+1 {{NPOT tt.reduce operations are not yet supported with modular tensor layouts}}
    %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.reduce.return %s : f32
    }) : (tensor<2x320xf32, #b>) -> tensor<2xf32, #ttg.slice<{dim = 1, parent = #b}>>
    tt.return
  }

  tt.func public @pot_control_2x256_axis1(%x: tensor<2x256xf32, #b>) {
    %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.reduce.return %s : f32
    }) : (tensor<2x256xf32, #b>) -> tensor<2xf32, #ttg.slice<{dim = 1, parent = #b}>>
    tt.return
  }
}
