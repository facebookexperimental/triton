// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// #p96 isn't exact at 64x192, nor #p192 at 64x96 (96 and 256, 192 and 128
// don't divide), so RLC must keep a convert between the reduce and the broadcast.
#p192 = #ttg.linear<{register = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [1, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 192]}>
#p96 = #ttg.linear<{register = [[4, 0], [8, 0], [16, 0], [32, 0], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[1, 0], [2, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 96]}>
// 1536 = 3 x 512: the load's coalesced #c1536 is a mixed-radix layout tt.scan
// allows, so RLC scans on it in place. The pow2 control #c1024 is not Blocked,
// so the converts around its scan stay.
#c1536 = #ttg.linear<{register = [[1], [2], [512]], lane = [[4], [8], [16], [32], [64]], warp = [[128], [256]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [1536]}>
#d1536 = #ttg.linear<{register = [[128], [256], [512]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [1536]}>
#c1024 = #ttg.linear<{register = [[1], [2], [512]], lane = [[4], [8], [16], [32], [64]], warp = [[128], [256]], block = []}>
#d1024 = #ttg.linear<{register = [[128], [256], [512]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @forward_onto_broadcast
  // CHECK: %[[R:.*]] = "tt.reduce"
  // CHECK: ttg.convert_layout %[[R]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P96:linear[0-9]*]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P192:linear[0-9]*]]}>>
  // CHECK: tt.broadcast %{{.*}} : tensor<64x1xf32, #[[P192]]> -> tensor<64x192xf32, #[[P192]]>
  tt.func public @forward_onto_broadcast(%p: !tt.ptr<f32>) {
    %pa = tt.splat %p : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>, #p96>
    %pb = tt.splat %p : !tt.ptr<f32> -> tensor<64x192x!tt.ptr<f32>, #p192>
    %a = tt.load %pa : tensor<64x96x!tt.ptr<f32>, #p96>
    %r = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %m = arith.maxnumf %x, %y : f32
      tt.reduce.return %m : f32
    }) : (tensor<64x96xf32, #p96>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>> -> tensor<64x1xf32, #p96>
    %c = ttg.convert_layout %e : tensor<64x1xf32, #p96> -> tensor<64x1xf32, #p192>
    %b = tt.broadcast %c : tensor<64x1xf32, #p192> -> tensor<64x192xf32, #p192>
    tt.store %pb, %b : tensor<64x192x!tt.ptr<f32>, #p192>
    tt.return
  }

  // CHECK-LABEL: @backward_into_reduce
  // CHECK: %[[R:.*]] = "tt.reduce"
  // CHECK: ttg.convert_layout %[[R]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P96:linear[0-9]*]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P192:linear[0-9]*]]}>>
  tt.func public @backward_into_reduce(%p: !tt.ptr<f32>, %x: f32) {
    %pb = tt.splat %p : !tt.ptr<f32> -> tensor<64x192x!tt.ptr<f32>, #p192>
    %a0 = tt.splat %x : f32 -> tensor<64x96xf32, #p96>
    %a = arith.mulf %a0, %a0 : tensor<64x96xf32, #p96>
    %r = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%u: f32, %v: f32):
      %m = arith.maxnumf %u, %v : f32
      tt.reduce.return %m : f32
    }) : (tensor<64x96xf32, #p96>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>>
    %c = ttg.convert_layout %r : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p192}>>
    %e = tt.expand_dims %c {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p192}>> -> tensor<64x1xf32, #p192>
    %b = tt.broadcast %e : tensor<64x1xf32, #p192> -> tensor<64x192xf32, #p192>
    tt.store %pb, %b : tensor<64x192x!tt.ptr<f32>, #p192>
    tt.return
  }

  // CHECK-LABEL: @scan_coalesced_mixed_radix
  // CHECK: tt.load %{{.*}} : tensor<1536x!tt.ptr<f32>, #[[C:linear[0-9]*]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: }) : (tensor<1536xf32, #[[C]]>) -> tensor<1536xf32, #[[C]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func public @scan_coalesced_mixed_radix(%p: !tt.ptr<f32>) {
    %pc = tt.splat %p : !tt.ptr<f32> -> tensor<1536x!tt.ptr<f32>, #c1536>
    %a = tt.load %pc : tensor<1536x!tt.ptr<f32>, #c1536>
    %b = ttg.convert_layout %a : tensor<1536xf32, #c1536> -> tensor<1536xf32, #d1536>
    %s = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.scan.return %t : f32
    }) : (tensor<1536xf32, #d1536>) -> tensor<1536xf32, #d1536>
    %c = ttg.convert_layout %s : tensor<1536xf32, #d1536> -> tensor<1536xf32, #c1536>
    tt.store %pc, %c : tensor<1536x!tt.ptr<f32>, #c1536>
    tt.return
  }

  // CHECK-LABEL: @scan_pow2_control
  // CHECK: ttg.convert_layout %{{.*}} : tensor<1024xf32, #[[C:linear[0-9]*]]> -> tensor<1024xf32, #[[D:linear[0-9]*]]>
  // CHECK: }) : (tensor<1024xf32, #[[D]]>) -> tensor<1024xf32, #[[D]]>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<1024xf32, #[[D]]> -> tensor<1024xf32, #[[C]]>
  tt.func public @scan_pow2_control(%p: !tt.ptr<f32>) {
    %pc = tt.splat %p : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #c1024>
    %a = tt.load %pc : tensor<1024x!tt.ptr<f32>, #c1024>
    %b = ttg.convert_layout %a : tensor<1024xf32, #c1024> -> tensor<1024xf32, #d1024>
    %s = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%x: f32, %y: f32):
      %t = arith.addf %x, %y : f32
      tt.scan.return %t : f32
    }) : (tensor<1024xf32, #d1024>) -> tensor<1024xf32, #d1024>
    %c = ttg.convert_layout %s : tensor<1024xf32, #d1024> -> tensor<1024xf32, #c1024>
    tt.store %pc, %c : tensor<1024x!tt.ptr<f32>, #c1024>
    tt.return
  }
}

// -----

// slice<0, #mma> is exact at 96 (the unit dim broadcasts #mma's M lanes and
// warps, as in pow2), so RLC does the column sum in #mma like the pow2 control.
#linear = #ttg.linear<{register = [[4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[1, 0], [2, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 96, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @colsum_mma_v3
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[D:.*]] = ttng.warp_group_dot
  // CHECK: "tt.reduce"(%[[D]])
  // CHECK: -> tensor<96xf32, #ttg.slice<{dim = 0, parent = #mma}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func public @colsum_mma_v3(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x96xf16, #shared1, #smem>, %pc: !tt.ptr<f32>) {
    %c = arith.constant dense<0.0> : tensor<128x96xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x96xf16, #shared1, #smem> -> tensor<128x96xf32, #mma>
    %e = ttg.convert_layout %d : tensor<128x96xf32, #mma> -> tensor<128x96xf32, #linear>
    %s = "tt.reduce"(%e) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %z = arith.addf %x, %y : f32
      tt.reduce.return %z : f32
    }) : (tensor<128x96xf32, #linear>) -> tensor<96xf32, #ttg.slice<{dim = 0, parent = #linear}>>
    %se = tt.expand_dims %s {axis = 0 : i32} : tensor<96xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x96xf32, #linear>
    %sb = tt.broadcast %se : tensor<1x96xf32, #linear> -> tensor<128x96xf32, #linear>
    %q = arith.divf %e, %sb : tensor<128x96xf32, #linear>
    %o = ttg.convert_layout %q : tensor<128x96xf32, #linear> -> tensor<128x96xf32, #mma>
    %pp = tt.splat %pc : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>, #mma>
    tt.store %pp, %o : tensor<128x96x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Pow2 control: the same column sum at 128x128.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @colsum_mma_v3_pow2
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[D:.*]] = ttng.warp_group_dot
  // CHECK: "tt.reduce"(%[[D]])
  // CHECK: -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #mma}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func public @colsum_mma_v3_pow2(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x128xf16, #shared1, #smem>, %pc: !tt.ptr<f32>) {
    %c = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<128x128xf32, #mma>
    %e = ttg.convert_layout %d : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %s = "tt.reduce"(%e) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %z = arith.addf %x, %y : f32
      tt.reduce.return %z : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %se = tt.expand_dims %s {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
    %sb = tt.broadcast %se : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %q = arith.divf %e, %sb : tensor<128x128xf32, #blocked>
    %o = ttg.convert_layout %q : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    %pp = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #mma>
    tt.store %pp, %o : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// #pow2 covers only 128x128, so RLC keeps the convert on the reduced rows and
// the 128x192 epilogue in #exact.
#pow2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#exact = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 192]}>
#store = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 64]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 192]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rescale_192
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[EXACT:linear1]]}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x192xf32, #[[EXACT]]>
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<128x192xf16, #[[EXACT]]>)
  tt.func public @rescale_192(%s: tensor<128x128xf32, #pow2>, %a: !tt.ptr<f32>) -> !ttg.memdesc<128x192xf16, #shared, #smem> {
    %l = "tt.reduce"(%s) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %r = arith.addf %x, %y : f32
      tt.reduce.return %r : f32
    }) : (tensor<128x128xf32, #pow2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #pow2}>>
    %pa = tt.splat %a : !tt.ptr<f32> -> tensor<128x192x!tt.ptr<f32>, #exact>
    %acc = tt.load %pa : tensor<128x192x!tt.ptr<f32>, #exact>
    %lc = ttg.convert_layout %l : tensor<128xf32, #ttg.slice<{dim = 1, parent = #pow2}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #exact}>>
    %le = tt.expand_dims %lc {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #exact}>> -> tensor<128x1xf32, #exact>
    %lb = tt.broadcast %le : tensor<128x1xf32, #exact> -> tensor<128x192xf32, #exact>
    %q = arith.divf %acc, %lb : tensor<128x192xf32, #exact>
    %h = arith.truncf %q : tensor<128x192xf32, #exact> to tensor<128x192xf16, #exact>
    %o = ttg.convert_layout %h : tensor<128x192xf16, #exact> -> tensor<128x192xf16, #store>
    %m = ttg.local_alloc %o : (tensor<128x192xf16, #store>) -> !ttg.memdesc<128x192xf16, #shared, #smem>
    tt.return %m : !ttg.memdesc<128x192xf16, #shared, #smem>
  }
}

// -----

// Pow2 control: at 128x256 the convert moves past the expand_dims.
#pow2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#wide = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#store = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 64], [0, 128]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0]], warp = [[4, 0], [8, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rescale_256
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #[[WIDE:linear1]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<128x256xf32, #[[WIDE]]>
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<128x256xf16, #[[WIDE]]>)
  tt.func public @rescale_256(%s: tensor<128x128xf32, #pow2>, %a: !tt.ptr<f32>) -> !ttg.memdesc<128x256xf16, #shared, #smem> {
    %l = "tt.reduce"(%s) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %r = arith.addf %x, %y : f32
      tt.reduce.return %r : f32
    }) : (tensor<128x128xf32, #pow2>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #pow2}>>
    %pa = tt.splat %a : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #wide>
    %acc = tt.load %pa : tensor<128x256x!tt.ptr<f32>, #wide>
    %lc = ttg.convert_layout %l : tensor<128xf32, #ttg.slice<{dim = 1, parent = #pow2}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #wide}>>
    %le = tt.expand_dims %lc {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #wide}>> -> tensor<128x1xf32, #wide>
    %lb = tt.broadcast %le : tensor<128x1xf32, #wide> -> tensor<128x256xf32, #wide>
    %q = arith.divf %acc, %lb : tensor<128x256xf32, #wide>
    %h = arith.truncf %q : tensor<128x256xf32, #wide> to tensor<128x256xf16, #wide>
    %o = ttg.convert_layout %h : tensor<128x256xf16, #wide> -> tensor<128x256xf16, #store>
    %m = ttg.local_alloc %o : (tensor<128x256xf16, #store>) -> !ttg.memdesc<128x256xf16, #shared, #smem>
    tt.return %m : !ttg.memdesc<128x256xf16, #shared, #smem>
  }
}
