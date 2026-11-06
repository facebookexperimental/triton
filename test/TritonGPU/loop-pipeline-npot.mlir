// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck --enable-var-scope %s

// A mixed-radix load pipelines through shared memory when its async copy and
// local_load lower on the target; otherwise it stays a plain load.

#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_pipelined
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x96xf32
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load {{.*}} -> tensor<96xf32, #linear>
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %step = arith.constant dense<96> : tensor<96xi32, #p96>
    %x = scf.for %i = %c0 to %n step %c1 iter_args(%q0 = %a) -> (tensor<96x!tt.ptr<f32>, #p96>) : i32 {
      %v = tt.load %q0 : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
      %q1 = tt.addptr %q0, %step : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
      scf.yield %q1 : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pow2_load_pipelined
  // CHECK: ttg.local_alloc
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  tt.func public @pow2_load_pipelined(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %a = tt.addptr %s, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %b = tt.addptr %t, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %b, %v : tensor<128x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// hip can't copy a mixed-radix layout asynchronously, so even a user latency on
// the load is dropped.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_user_latency
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.for
  // CHECK-NEXT: tt.load
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_user_latency(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a {tt.latency = 2 : i32} : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// sm80 reads the stage back through a mixed-radix local_load, which takes the
// convert's layout.
#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_convert_sm80
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load {{.*}} -> tensor<192xf32, #linear1>
  // CHECK-NEXT: tt.store
  tt.func public @mixed_radix_load_convert_sm80(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #src>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<f32>, #src>, tensor<192xi32, #src>
    %rd = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #dst>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<192x!tt.ptr<f32>, #dst>
    %b = tt.addptr %t, %rd : tensor<192x!tt.ptr<f32>, #dst>, tensor<192xi32, #dst>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a : tensor<192x!tt.ptr<f32>, #src>
      %w = ttg.convert_layout %v : tensor<192xf32, #src> -> tensor<192xf32, #dst>
      tt.store %b, %w : tensor<192x!tt.ptr<f32>, #dst>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// A non-zero `other` is selected after the mixed-radix local_load.
#p96 = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_load_other
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.local_load
  // CHECK-NEXT: arith.select
  tt.func public @mixed_radix_load_other(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %q: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32, %m: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<96xf32, #p96>
    %r = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #p96>
    %ms = tt.splat %m : i32 -> tensor<96xi32, #p96>
    %k = arith.cmpi slt, %r, %ms : tensor<96xi32, #p96>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %a = tt.addptr %s, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    %t = tt.splat %q : !tt.ptr<f32> -> tensor<96x!tt.ptr<f32>, #p96>
    %b = tt.addptr %t, %r : tensor<96x!tt.ptr<f32>, #p96>, tensor<96xi32, #p96>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %a, %k, %one : tensor<96x!tt.ptr<f32>, #p96>
      tt.store %b, %v : tensor<96x!tt.ptr<f32>, #p96>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// sm90 copies a mixed-radix B operand of a wgmma into a 3-stage buffer.
#a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#b = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 96, 16]}>
#nv = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_mixed_radix_b
  // CHECK-DAG: ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16
  // CHECK-DAG: ttg.local_alloc : () -> !ttg.memdesc<3x32x96xf16
  // CHECK: scf.for
  // CHECK: ttng.warp_group_dot
  // CHECK-COUNT-2: ttg.async_copy_global_to_local
  // CHECK: scf.yield
  tt.func public @wgmma_mixed_radix_b(%pa: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %po: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<128x96xf32, #mma>
    %ra = arith.constant dense<1024> : tensor<128x1xi32, #a>
    %sa = arith.constant dense<32> : tensor<128x32xi32, #a>
    %rb = arith.constant dense<96> : tensor<32x1xi32, #b>
    %sb = arith.constant dense<3072> : tensor<32x96xi32, #b>
    %am = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #a}>>
    %am1 = tt.expand_dims %am {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #a}>> -> tensor<128x1xi32, #a>
    %am2 = arith.muli %am1, %ra : tensor<128x1xi32, #a>
    %am3 = tt.broadcast %am2 : tensor<128x1xi32, #a> -> tensor<128x32xi32, #a>
    %ak = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #a}>>
    %ak1 = tt.expand_dims %ak {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #a}>> -> tensor<1x32xi32, #a>
    %ak2 = tt.broadcast %ak1 : tensor<1x32xi32, #a> -> tensor<128x32xi32, #a>
    %ao = arith.addi %am3, %ak2 : tensor<128x32xi32, #a>
    %as = tt.splat %pa : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #a>
    %a0 = tt.addptr %as, %ao : tensor<128x32x!tt.ptr<f16>, #a>, tensor<128x32xi32, #a>
    %bk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #b}>>
    %bk1 = tt.expand_dims %bk {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #b}>> -> tensor<32x1xi32, #b>
    %bk2 = arith.muli %bk1, %rb : tensor<32x1xi32, #b>
    %bk3 = tt.broadcast %bk2 : tensor<32x1xi32, #b> -> tensor<32x96xi32, #b>
    %bn = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #ttg.slice<{dim = 0, parent = #b}>>
    %bn1 = tt.expand_dims %bn {axis = 0 : i32} : tensor<96xi32, #ttg.slice<{dim = 0, parent = #b}>> -> tensor<1x96xi32, #b>
    %bn2 = tt.broadcast %bn1 : tensor<1x96xi32, #b> -> tensor<32x96xi32, #b>
    %bo = arith.addi %bk3, %bn2 : tensor<32x96xi32, #b>
    %bs = tt.splat %pb : !tt.ptr<f16> -> tensor<32x96x!tt.ptr<f16>, #b>
    %b0 = tt.addptr %bs, %bo : tensor<32x96x!tt.ptr<f16>, #b>, tensor<32x96xi32, #b>
    %r:3 = scf.for %i = %c0 to %n step %c32 iter_args(%acc = %acc0, %qa = %a0, %qb = %b0) -> (tensor<128x96xf32, #mma>, tensor<128x32x!tt.ptr<f16>, #a>, tensor<32x96x!tt.ptr<f16>, #b>) : i32 {
      %av = tt.load %qa : tensor<128x32x!tt.ptr<f16>, #a>
      %am_s = ttg.local_alloc %av : (tensor<128x32xf16, #a>) -> !ttg.memdesc<128x32xf16, #nv, #smem>
      %bv = tt.load %qb : tensor<32x96x!tt.ptr<f16>, #b>
      %bm_s = ttg.local_alloc %bv : (tensor<32x96xf16, #b>) -> !ttg.memdesc<32x96xf16, #nv, #smem>
      %d = ttng.warp_group_dot %am_s, %bm_s, %acc : !ttg.memdesc<128x32xf16, #nv, #smem> * !ttg.memdesc<32x96xf16, #nv, #smem> -> tensor<128x96xf32, #mma>
      %na = tt.addptr %qa, %sa : tensor<128x32x!tt.ptr<f16>, #a>, tensor<128x32xi32, #a>
      %nb = tt.addptr %qb, %sb : tensor<32x96x!tt.ptr<f16>, #b>, tensor<32x96xi32, #b>
      scf.yield %d, %na, %nb : tensor<128x96xf32, #mma>, tensor<128x32x!tt.ptr<f16>, #a>, tensor<32x96x!tt.ptr<f16>, #b>
    } {tt.num_stages = 3 : i32}
    %ps = tt.splat %po : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>, #mma>
    tt.store %ps, %r#0 : tensor<128x96x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// sm90 keeps a pipelined mixed-radix A RS wgmma whole: splitting K goes through
// tt.split, whose layout inference is binary-only. The pow2 control splits.
#b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#nv = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wgmma_rs_split
  // CHECK: scf.for
  // CHECK-NOT: tt.split
  // CHECK: ttng.warp_group_dot {{.*}} : tensor<192x32xf16
  // CHECK-NOT: ttng.warp_group_dot {{.*}} : tensor
  // CHECK: scf.yield
  // CHECK: scf.for
  // CHECK: tt.split
  // CHECK-COUNT-2: ttng.warp_group_dot {{.*}} : tensor<128x16xf16
  tt.func public @wgmma_rs_split(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %po: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %x0 = arith.constant dense<0.000000e+00> : tensor<192x128xf32, #mma>
    %y0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %xa = arith.constant dense<1.000000e+00> : tensor<192x32xf16, #dA>
    %ya = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dA>
    %rb = arith.constant dense<128> : tensor<32x1xi32, #b>
    %sb = arith.constant dense<4096> : tensor<32x128xi32, #b>
    %bk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #b}>>
    %bk1 = tt.expand_dims %bk {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #b}>> -> tensor<32x1xi32, #b>
    %bk2 = arith.muli %bk1, %rb : tensor<32x1xi32, #b>
    %bk3 = tt.broadcast %bk2 : tensor<32x1xi32, #b> -> tensor<32x128xi32, #b>
    %bn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #b}>>
    %bn1 = tt.expand_dims %bn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #b}>> -> tensor<1x128xi32, #b>
    %bn2 = tt.broadcast %bn1 : tensor<1x128xi32, #b> -> tensor<32x128xi32, #b>
    %bo = arith.addi %bk3, %bn2 : tensor<32x128xi32, #b>
    %bs = tt.splat %pb : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #b>
    %b0 = tt.addptr %bs, %bo : tensor<32x128x!tt.ptr<f16>, #b>, tensor<32x128xi32, #b>
    %x:2 = scf.for %i = %c0 to %n step %c32 iter_args(%acc = %x0, %qb = %b0) -> (tensor<192x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #b>) : i32 {
      %bv = tt.load %qb : tensor<32x128x!tt.ptr<f16>, #b>
      %bm = ttg.local_alloc %bv : (tensor<32x128xf16, #b>) -> !ttg.memdesc<32x128xf16, #nv, #smem>
      %d = ttng.warp_group_dot %xa, %bm, %acc : tensor<192x32xf16, #dA> * !ttg.memdesc<32x128xf16, #nv, #smem> -> tensor<192x128xf32, #mma>
      %nb = tt.addptr %qb, %sb : tensor<32x128x!tt.ptr<f16>, #b>, tensor<32x128xi32, #b>
      scf.yield %d, %nb : tensor<192x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #b>
    } {tt.num_stages = 3 : i32}
    %y:2 = scf.for %i = %c0 to %n step %c32 iter_args(%acc = %y0, %qb = %b0) -> (tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #b>) : i32 {
      %bv = tt.load %qb : tensor<32x128x!tt.ptr<f16>, #b>
      %bm = ttg.local_alloc %bv : (tensor<32x128xf16, #b>) -> !ttg.memdesc<32x128xf16, #nv, #smem>
      %d = ttng.warp_group_dot %ya, %bm, %acc : tensor<128x32xf16, #dA> * !ttg.memdesc<32x128xf16, #nv, #smem> -> tensor<128x128xf32, #mma>
      %nb = tt.addptr %qb, %sb : tensor<32x128x!tt.ptr<f16>, #b>, tensor<32x128xi32, #b>
      scf.yield %d, %nb : tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #b>
    } {tt.num_stages = 3 : i32}
    %px = tt.splat %po : !tt.ptr<f32> -> tensor<192x128x!tt.ptr<f32>, #mma>
    tt.store %px, %x#0 : tensor<192x128x!tt.ptr<f32>, #mma>
    %py = tt.splat %po : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #mma>
    tt.store %py, %y#0 : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
