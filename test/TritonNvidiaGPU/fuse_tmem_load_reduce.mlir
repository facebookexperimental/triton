// RUN: triton-opt %s -split-input-file --triton-nvidia-tmem-load-reduce --allow-unregistered-dialect | FileCheck %s

// Fuse tmem_load + tt.reduce -> tmem_load, with redOp=max. The combiner
// for "arith.maxnumf" ignores NaNs, so the fused op does not set the NaN
// attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_max(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {redOp = #ttng.redOp<max>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   tt.return %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_max(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Fuse (tmem_load, tt.reduce) -> tmem_load, with redOp=min
// and NaN=true, and arith.minimumf is the NaN-propagating variant.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_min_nan(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {NaN = true, redOp = #ttng.redOp<min>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   tt.return %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_min_nan(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.minimumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Look through "ttg.convert_layout" that sits between the "tmem_load" and the
// "tt.reduce" and check that the combine happens.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_through_cvt(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {NaN = true, redOp = #ttng.redOp<max>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %[[CVT:.+]] = ttg.convert_layout %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT:   tt.return %[[CVT]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_through_cvt(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %cvt = ttg.convert_layout %0 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #linear>
    %1 = "tt.reduce"(%cvt) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  }
}

// -----

// Negative test: an explicitly ordered reduction must preserve its reduction
// tree and must not be fused into tmem_load.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_inner_tree
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  // CHECK-SAME: reduction_ordering = "inner_tree"
  tt.func public @tmem_load_reduce_no_fuse_inner_tree(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32, reduction_ordering = "inner_tree"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: check that that we do not fuse and generate "tcgen05.ld.red" if
// the target isn't sm103+.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_sm100
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_sm100(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: sm120/sm121 are explicitly excluded by supportLdRed() even
// though they are >= sm103, so the pass must leave "tt.reduce" unfused.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_sm120
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_sm120(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: "arith.addf" is not a supported combiner.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_addf
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_addf(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.addf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test and current Ttriton limitation: make sure we don't
// fuse when the element is a i32 as the reduction on operation tmem_load
// is documented as f32-only.
// Even with a max/min-style integer combiner on sm103, the pattern must bail.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_i32
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_i32(%arg0: !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory> -> tensor<128x128xi32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %2 = arith.maxsi %lhs, %rhs : i32
      tt.reduce.return %2 : i32
    }) : (tensor<128x128xi32, #blocked>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: reduction along axis 0 (the M axis) must not fuse. The
// fused tcgen05.ld.red only reduces along the inner N axis.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_axis0
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_axis0(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}

// -----

// Negative test for the "already fused" bail-out.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_already_fused
  // CHECK: %{{.+}}, %{{.+}} = ttng.tmem_load{{.+}}redOp = #ttng.redOp<max>
  // CHECK-NOT: ttng.tmem_load
  // CHECK: "tt.reduce"
  // CHECK-NOT: "tt.reduce"
  // CHECK: tt.return
  tt.func public @tmem_load_reduce_no_fuse_already_fused(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %a = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %a : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %b = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %b : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1, %2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: the layout below splits the N dimension across registers and warps,
// 2 warps along N, so N is not entirely in register.

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_n_sharded_across_warps
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_n_sharded_across_warps(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Warp-spec partition boundary: tmem_load followed by pure barrier-slot
// bookkeeping and arrive before the row reduction. The pass must hoist the
// arrive past the reduction (no intervening memory ops) and fuse to
// tcgen05.ld.red. This covers the AutoWS memdesc_index and TLX explicit
// `tlx.barrier_arrive` cases at the partition edge.

#blocked_ws = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_ws = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared_ws = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_with_arrive(
  // CHECK-SAME:    %[[TMEM:.+]]: !ttg.memdesc<128x128xf32, #{{.*}}, #ttng.tensor_memory>
  // CHECK-SAME:    %[[BARS:.+]]: !ttg.memdesc<2x1xi64, #{{.*}}, #{{.*}}, mutable>
  // CHECK-SAME:    %[[INDEX:.+]]: i32
  // CHECK:  %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[TMEM]] {{.*}}redOp = #ttng.redOp<max>
  // CHECK-NEXT:  %[[BAR:.+]] = ttg.memdesc_index %[[BARS]][%[[INDEX]]]
  // CHECK-NEXT:  ttng.arrive_barrier %[[BAR]]
  // CHECK-NEXT:  tt.return %[[RED]]
  tt.func public @tmem_load_reduce_fuse_with_arrive(%arg0: !ttg.memdesc<128x128xf32, #tmem_ws, #ttng.tensor_memory>, %bars: !ttg.memdesc<2x1xi64, #shared_ws, #ttg.shared_memory, mutable>, %index: i32) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem_ws, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked_ws>
    %bar = ttg.memdesc_index %bars[%index] : !ttg.memdesc<2x1xi64, #shared_ws, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_ws, #ttg.shared_memory, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_ws, #ttg.shared_memory, mutable>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked_ws>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws}>>
  }
}

// -----

// Negative: an otherwise pure operation that consumes the tmem_load result
// is not independent and must still block fusion.

#blocked_ws_dep = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_ws_dep = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared_ws_dep = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_with_dependent_pure_op
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: arith.negf
  // CHECK: ttng.arrive_barrier
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_with_dependent_pure_op(%arg0: !ttg.memdesc<128x128xf32, #tmem_ws_dep, #ttng.tensor_memory>, %bar: !ttg.memdesc<1xi64, #shared_ws_dep, #ttg.shared_memory, mutable>) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws_dep}>>, tensor<128x128xf32, #blocked_ws_dep>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem_ws_dep, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked_ws_dep>
    %unused = arith.negf %0 : tensor<128x128xf32, #blocked_ws_dep>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_ws_dep, #ttg.shared_memory, mutable>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked_ws_dep>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws_dep}>>
    tt.return %1, %unused : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws_dep}>>, tensor<128x128xf32, #blocked_ws_dep>
  }
}

// -----

// Same as above but with an intervening convert_layout (must still fuse) and
// a barrier between load and reduce. Also verifies per-subtile behavior.

#blocked_ws2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear_ws = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem_ws2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared_ws2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_with_arrive_through_cvt(
  // CHECK:  %{{.+}}, %[[RED:.+]] = ttng.tmem_load {{.*}}redOp = #ttng.redOp<max>
  // CHECK-NEXT:  ttg.convert_layout %[[RED]]
  // CHECK-NEXT:  ttng.arrive_barrier
  tt.func public @tmem_load_reduce_fuse_with_arrive_through_cvt(%arg0: !ttg.memdesc<128x128xf32, #tmem_ws2, #ttng.tensor_memory>, %bar: !ttg.memdesc<1xi64, #shared_ws2, #ttg.shared_memory, mutable>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear_ws}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem_ws2, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked_ws2>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_ws2, #ttg.shared_memory, mutable>
    %cvt = ttg.convert_layout %0 : tensor<128x128xf32, #blocked_ws2> -> tensor<128x128xf32, #linear_ws>
    %1 = "tt.reduce"(%cvt) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #linear_ws>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear_ws}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear_ws}>>
  }
}

// -----

// Negative: if a real memory op sits between tmem_load+arrive and the
// reduction, the barrier cannot be hoisted and fusion must not occur.

#blocked_ws3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_ws3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared_ws3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_with_mem_op_between
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: ttng.arrive_barrier
  // CHECK: tt.load
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_with_mem_op_between(%arg0: !ttg.memdesc<128x128xf32, #tmem_ws3, #ttng.tensor_memory>, %bar: !ttg.memdesc<1xi64, #shared_ws3, #ttg.shared_memory, mutable>, %ptr: !tt.ptr<f32>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws3}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem_ws3, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked_ws3>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_ws3, #ttg.shared_memory, mutable>
    %tmp = tt.load %ptr : !tt.ptr<f32>
    %tmp2 = arith.addf %tmp, %tmp : f32
    tt.store %ptr, %tmp2 : !tt.ptr<f32>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked_ws3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws3}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_ws3}>>
  }
}
