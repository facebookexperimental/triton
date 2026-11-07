// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

// Under TRITON_ALLOW_NPOT only TTIR tensors and exact mixed-radix layouts may
// be NPOT. Every other encoded tensor keeps the power-of-two rule, including
// block arguments and results of ops without the tensor-size trait.

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @arg_96(%x: tensor<96xf32, #b>) {
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @constant_96() {
    // expected-error @+1 {{Number of elements must be power-of-two, but}}
    %c = arith.constant dense<0.0> : tensor<96xf32, #b>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @func_num_warps_arg_96(%x: tensor<96xf32, #b>) attributes {"ttg.num-warps" = 4 : i32} {
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ttir_96_control(%x: tensor<96xf32>, %f: f32) -> tensor<96xf32> {
    %s = tt.splat %f : f32 -> tensor<96xf32>
    %c = arith.constant dense<1.0> : tensor<96xf32>
    %y = arith.addf %s, %c : tensor<96xf32>
    tt.return %y : tensor<96xf32>
  }
}

// -----

// An exact mixed-radix layout verifies on the ops that lower it.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#q96 = #ttg.linear<{register = [[32]], lane = [[2], [1], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_96(%p: !tt.ptr<f32>) {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %v = tt.load %a : tensor<96x!tt.ptr<f32>, #p96>
    %w = arith.addf %v, %v : tensor<96xf32, #p96>
    %u = ub.poison : tensor<96xf32, #p96>
    %c = ttg.convert_layout %w : tensor<96xf32, #p96> -> tensor<96xf32, #q96>
    %b = ttg.convert_layout %a : tensor<96x!tt.ptr<f32>, #p96> -> tensor<96x!tt.ptr<f32>, #q96>
    tt.store %b, %c : tensor<96x!tt.ptr<f32>, #q96>
    tt.return
  }
}

// -----

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
  tt.func @mixed_radix_arg(%x: tensor<96xf32, #p96>) {
    // OFF: [[@LINE+1]]:10: error: Operand 0 {{.*}} has an invalid layout: Layout has shape 96, but the tensor it's attached to has shape 96 which is not a power of two.
    %c = ttg.convert_layout %x : tensor<96xf32, #p96> -> tensor<96xf32, #p96>
    tt.return
  }
}

// -----

// An op without MixedRadixLayoutTrait or MixedRadixLayoutOpInterface is
// rejected.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_map_elementwise() {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.map_elementwise}}
    %m = "tt.map_elementwise"(%r) <{pack = 1 : i32}> ({
    ^bb0(%a: i32):
      tt.map_elementwise.return %a : i32
    }) : (tensor<96xi32, #p96>) -> tensor<96xi32, #p96>
    tt.return
  }
}

// -----

// A scan takes a mixed-radix layout unless it names why its lowering can't yet.
// [96, 80] appends radix 3 on 96, then radix 5 on 80.
#q = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [0, 16]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 3, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 80]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_scan() {
    %x = arith.constant dense<1> : tensor<96x80xi32, #q>
    %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<96x80xi32, #q>) -> tensor<96x80xi32, #q>
    %u = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<96x80xi32, #q>) -> tensor<96x80xi32, #q>
    tt.return
  }
}

// -----

// #q with its second lane bit moved to dim 0: dim 1's lanes are no longer
// consecutive, so its lanes don't step as a blocked layout's do.
#qi = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [0, 16]], lane = [[0, 4], [1, 0], [0, 8], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 3, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 80]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_scan_unblocked_lanes() {
    %x = arith.constant dense<1> : tensor<96x80xi32, #qi>
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on a scan whose lanes or warps are not blocked}}
    %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<96x80xi32, #qi>) -> tensor<96x80xi32, #qi>
    tt.return
  }
}

// -----

#q = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [0, 16]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 3, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 80]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_scan_inner_tree() {
    %x = arith.constant dense<1> : tensor<96x80xi32, #q>
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on an inner_tree scan}}
    %v = "tt.scan"(%x) <{axis = 1 : i32, reverse = false, reduction_ordering = "inner_tree"}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<96x80xi32, #q>) -> tensor<96x80xi32, #q>
    tt.return
  }
}

// -----

// A reduce takes a mixed-radix layout unless it names why its lowering can't
// yet.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_reduce_inner_tree() {
    %x = arith.constant dense<1.0> : tensor<32x96xf32, #p>
    %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<32x96xf32, #p>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #p}>>
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on an inner_tree reduce}}
    %t = "tt.reduce"(%x) <{axis = 1 : i32, reduction_ordering = "inner_tree"}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<32x96xf32, #p>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #p}>>
    tt.return
  }
}

// -----

// A used atomic result may not have duplicates: #pl broadcasts a lane bit.
#pl = #ttg.linear<{register = [[0, 1], [0, 2], [8, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [0, 0]], warp = [[2, 0], [4, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [16, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_atomics(%p: !tt.ptr<i32>) {
    %pl = tt.splat %p : !tt.ptr<i32> -> tensor<16x96x!tt.ptr<i32>, #pl>
    %vl = arith.constant dense<1> : tensor<16x96xi32, #pl>
    // expected-error @+1 {{NPOT mixed-radix layout with broadcast elements is not yet supported on an atomic whose result is used}}
    %c = tt.atomic_cas acq_rel, gpu, %pl, %vl, %vl : (tensor<16x96x!tt.ptr<i32>, #pl>, tensor<16x96xi32, #pl>, tensor<16x96xi32, #pl>) -> tensor<16x96xi32, #pl>
    tt.store %pl, %c : tensor<16x96x!tt.ptr<i32>, #pl>
    tt.return
  }
}

// -----

// Constants, carried values, print, assert and histogram take a mixed-radix
// layout.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_free_ops(%n: i32, %c: i1) {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %two = arith.constant dense<2> : tensor<96xi32, #p96>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %r) -> (tensor<96xi32, #p96>) : i32 {
      %m = arith.muli %acc, %two : tensor<96xi32, #p96>
      scf.yield %m : tensor<96xi32, #p96>
    }
    %if = scf.if %c -> (tensor<96xi32, #p96>) {
      scf.yield %f : tensor<96xi32, #p96>
    } else {
      scf.yield %r : tensor<96xi32, #p96>
    }
    %w = scf.while (%a = %if) : (tensor<96xi32, #p96>) -> tensor<96xi32, #p96> {
      scf.condition(%c) %a : tensor<96xi32, #p96>
    } do {
    ^bb0(%a: tensor<96xi32, #p96>):
      scf.yield %a : tensor<96xi32, #p96>
    }
    cf.cond_br %c, ^bb1(%w : tensor<96xi32, #p96>), ^bb1(%r : tensor<96xi32, #p96>)
  ^bb1(%x: tensor<96xi32, #p96>):
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %x : tensor<96xi32, #p96>
    %pos = arith.cmpi sge, %x, %r : tensor<96xi32, #p96>
    tt.assert %pos, "x >= r" : tensor<96xi1, #p96>
    %h = tt.histogram %x : tensor<96xi32, #p96> -> tensor<16xi32, #b>
    cf.br ^bb2
  ^bb2:
    tt.return
  }
}

// -----

// A mixed-radix layout of another shape is not exact.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_layout_on_192() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.make_range}}
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #p96>
    tt.return
  }
}

// -----

// A layout is exact where toLinearLayout covers each out dim exactly: a slice
// of a mixed-radix layout, and a mixed-radix dim laid out at extent 1.
#p = #ttg.linear<{register = [[32, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = [], inputRadices = {register = [3, 2, 2, 2, 2, 2, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96, 64]}>
#q = #ttg.linear<{register = [[0, 32], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = [], inputRadices = {register = [3, 2, 2, 2, 2, 2, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [64, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_slice_and_unit_dim(%x: i32) {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #ttg.slice<{dim = 1, parent = #p}>>
    %m = tt.splat %x : i32 -> tensor<96x1xi32, #p>
    %n = tt.splat %x : i32 -> tensor<64x1xi32, #q>
    tt.return
  }
}

// -----

// Not exact: 96 does not shrink to the pow2 ceiling of 48.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_layout_on_48() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.make_range}}
    %r = tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32, #p96>
    tt.return
  }
}

// -----

// Not exact: 96 laid out at 32 does not cover it.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_layout_on_32() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.make_range}}
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #p96>
    tt.return
  }
}

// -----

// Not exact: a non-binary digit owns no odd part.
#f96 = #ttg.linear<{register = [[32], [0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_free_digit() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.make_range}}
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #f96>
    tt.return
  }
}

// -----

// Lane, warp and block digits stay binary.
#w96 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [], lane = [2, 2, 2, 2, 2], warp = [3], block = []}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 3 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_odd_warp() {
    // expected-error @+1 {{NPOT mixed-radix layout with a non-binary warp digit is not yet supported}}
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #w96>
    tt.return
  }
}

// -----

// Index math is 32-bit, so the odd part is bounded.
#big = #ttg.linear<{register = [[8]], lane = [[1], [2], [4], [0], [0]], warp = [], block = [], inputRadices = {register = [65537], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [524296]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_odd_part_above_65535() {
    // expected-error @+1 {{NPOT mixed-radix layout with an odd part above 65535 is not yet supported}}
    %r = tt.make_range {end = 524296 : i32, start = 0 : i32} : tensor<524296xi32, #big>
    tt.return
  }
}

// -----

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [[0]], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = [2]}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_two_ctas() {
    // expected-error @+1 {{NPOT mixed-radix layout across CTAs is not yet supported}}
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    tt.return
  }
}

// -----

// A convert is accepted iff its mixed-radix plan is supported.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
#q96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [[0]], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = [2]}, outDimSizes = [96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_to_cross_cta() {
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    // expected-error @+1 {{NPOT mixed-radix conversion with a non-singleton block input in the destination layout is not yet supported}}
    %c = ttg.convert_layout %r : tensor<96xi32, #p96> -> tensor<96xi32, #q96>
    tt.return
  }
}

// -----

// Planning builds both layouts and can assert on a non-mixed-radix side.
#p = #ttg.linear<{register = [[4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[1, 0], [2, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#b = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_to_fma_operand() {
    %a = arith.constant dense<0.000000e+00> : tensor<128x96xf16, #p>
    // expected-error @+1 {{NPOT conversion to or from a non-mixed-radix layout is not yet supported}}
    %c = ttg.convert_layout %a : tensor<128x96xf16, #p> -> tensor<128x96xf16, #ttg.dot_op<{opIdx = 0, parent = #b}>>
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_shared_force_warp_shuffle() {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    // expected-error @+1 {{NPOT mixed-radix conversion through shared memory is not yet supported with force_warp_shuffle}}
    %c = ttg.convert_layout %r {force_warp_shuffle} : tensor<192xi32, #src> -> tensor<192xi32, #dst>
    tt.return
  }
}

// -----

#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_pow2_alloc_view(%d: !ttg.memdesc<128x96xf16, #s8, #smem, mutable, 128x128>) {
    // expected-error @+1 {{NPOT mixed-radix access to a padded, partitioned or partial buffer is not yet supported}}
    %v = ttg.local_load %d : !ttg.memdesc<128x96xf16, #s8, #smem, mutable, 128x128> -> tensor<128x96xf16, #p>
    tt.return
  }
}

// -----

// An MMAv2 layout at an NPOT shape is an exact mixed-radix layout, so its dot
// verifies.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mma_v2_128x96() {
    %a = arith.constant dense<1.0> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %b = arith.constant dense<1.0> : tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %c = arith.constant dense<0.0> : tensor<128x96xf32, #mma>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x96xf32, #mma>
    // Control: the pow2 dot verifies.
    %b2 = arith.constant dense<1.0> : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %c2 = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %d2 = tt.dot %a, %b2, %c2 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
