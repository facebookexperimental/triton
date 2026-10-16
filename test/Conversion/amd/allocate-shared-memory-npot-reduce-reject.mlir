// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory=arch=gfx950 -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-warp-pipeline=gfx-arch=gfx950 -verify-diagnostics

// AMD twin of test/Conversion/allocate-shared-memory-npot-reduce-reject.mlir.
// convert-warp-pipeline builds a ModuleAllocation before allocation, so it
// rejects too.

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

  tt.func public @max_2x320_axis1(%x: tensor<2x320xf32, #b>) {
    // expected-error @+1 {{NPOT tt.reduce operations are not yet supported with modular tensor layouts}}
    %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.maxnumf %a, %c : f32
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

// -----

#b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k0_32x40_axis1(%x: tensor<32x40xf32, #b>) {
    // expected-error @+1 {{NPOT tt.reduce operations are not yet supported with modular tensor layouts}}
    %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.reduce.return %s : f32
    }) : (tensor<32x40xf32, #b>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #b}>>
    tt.return
  }

  tt.func public @k5_40x32_axis0(%x: tensor<40x32xf32, #b>) {
    // expected-error @+1 {{NPOT tt.reduce operations are not yet supported with modular tensor layouts}}
    %y = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.reduce.return %s : f32
    }) : (tensor<40x32xf32, #b>) -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #b}>>
    tt.return
  }

  tt.func public @pot_axis_40x32_axis1(%x: tensor<40x32xf32, #b>) {
    // expected-error @+1 {{NPOT tt.reduce operations are not yet supported with modular tensor layouts}}
    %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.reduce.return %s : f32
    }) : (tensor<40x32xf32, #b>) -> tensor<40xf32, #ttg.slice<{dim = 1, parent = #b}>>
    tt.return
  }

  tt.func public @gather_96x32_axis0(%src: tensor<96x32xf32, #b>, %idx: tensor<96x32xi32, #b>) {
    // expected-error @+1 {{NPOT tt.gather operations are not yet supported with modular tensor layouts}}
    %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<96x32xf32, #b>, tensor<96x32xi32, #b>) -> tensor<96x32xf32, #b>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scan_96(%x: tensor<96xf32, #b>) {
    // expected-error @+1 {{NPOT tt.scan operations are not yet supported with modular tensor layouts}}
    %y = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.scan.return %s : f32
    }) : (tensor<96xf32, #b>) -> tensor<96xf32, #b>
    tt.return
  }

  tt.func public @histogram_96_to_16(%x: tensor<96xi32, #b>) {
    // expected-error @+1 {{NPOT tt.histogram operations are not yet supported with modular tensor layouts}}
    %h = tt.histogram %x : tensor<96xi32, #b> -> tensor<16xi32, #b>
    tt.return
  }

  tt.func public @histogram_64_to_24(%x: tensor<64xi32, #b>) {
    // No diagnostic: NPOT bins over a power-of-two input are exact.
    %h = tt.histogram %x : tensor<64xi32, #b> -> tensor<24xi32, #b>
    tt.return
  }
}
