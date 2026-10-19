// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory -verify-diagnostics

// Under TRITON_ALLOW_NPOT, allocation rejects NPOT tt.reduce, tt.scan and
// tt.gather, and tt.histogram with an NPOT input, before ModuleAllocation.

#b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#g = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#b1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
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

  tt.func public @gather_96x32_axis0(%src: tensor<96x32xf32, #g>, %idx: tensor<96x32xi32, #g>) {
    // expected-error @+1 {{NPOT tt.gather operations are not yet supported with modular tensor layouts}}
    %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<96x32xf32, #g>, tensor<96x32xi32, #g>) -> tensor<96x32xf32, #g>
    tt.return
  }

  tt.func public @scan_96(%x: tensor<96xf32, #b1>) {
    // expected-error @+1 {{NPOT tt.scan operations are not yet supported with modular tensor layouts}}
    %y = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %c: f32):
      %s = arith.addf %a, %c : f32
      tt.scan.return %s : f32
    }) : (tensor<96xf32, #b1>) -> tensor<96xf32, #b1>
    tt.return
  }

  tt.func public @histogram_96_to_16(%x: tensor<96xi32, #b1>) {
    // expected-error @+1 {{NPOT tt.histogram operations are not yet supported with modular tensor layouts}}
    %h = tt.histogram %x : tensor<96xi32, #b1> -> tensor<16xi32, #b1>
    tt.return
  }

  tt.func public @histogram_64_to_24(%x: tensor<64xi32, #b1>) {
    // No diagnostic: NPOT bins over a power-of-two input are exact.
    %h = tt.histogram %x : tensor<64xi32, #b1> -> tensor<24xi32, #b1>
    tt.return
  }
}
