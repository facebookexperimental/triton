// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=3 -verify-diagnostics | FileCheck %s

// A reduction over the partitioned dim (a column sum of an M-partitioned
// accumulator) leaves each partition with a partial result over its rows.
// The partials are combined with the reduction's combiner, and the store reads
// the combined result.

// CHECK-LABEL: @col_reduce_combines_partials
// CHECK: [[P0:%.*]] = "tt.reduce"({{.*}}) <{axis = 0 : i32{{.*}}}>
// CHECK: (tensor<64x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: [[P1:%.*]] = "tt.reduce"({{.*}}) <{axis = 0 : i32{{.*}}}>
// CHECK: (tensor<64x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: [[SUM:%.*]] = arith.addf [[P0]], [[P1]] {{.*}}: tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: tt.store {{.*}}, [[SUM]] {{.*}}:
// CHECK-NOT: tensor<128x256xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @col_reduce_combines_partials(%desc_a: !tt.tensordesc<128x64xf16>, %desc_b: !tt.tensordesc<64x256xf16>, %out: !tt.ptr<f32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %sum = "tt.reduce"(%dot) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<128x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %ptr = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.store %ptr, %sum {async_task_id = array<i32: 1, 2>} : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.return
  }
}

// -----

// A multi-result combiner (argmax) is applied elementwise to the partials, and
// a tile op that reads the combined result through a broadcast (y - colmax)
// stays partitioned.

// CHECK-LABEL: @col_argmax_feeds_partitioned_tile
// CHECK: tt.make_range {{.*}}end = 64 : i32, start = 0 : i32
// CHECK: [[V0:%.*]]:2 = "tt.reduce"
// CHECK: tt.make_range {{.*}}end = 128 : i32, start = 64 : i32
// CHECK: [[V1:%.*]]:2 = "tt.reduce"
// CHECK: [[GT:%.*]] = arith.cmpf ogt, [[V0]]#0, [[V1]]#0 {{.*}}: tensor<256xf32
// CHECK: [[MAX:%.*]] = arith.select [[GT]], [[V0]]#0, [[V1]]#0 {{.*}}: tensor<256xi1
// CHECK: [[IDX:%.*]] = arith.select [[GT]], [[V0]]#1, [[V1]]#1 {{.*}}: tensor<256xi1
// CHECK: tt.expand_dims [[MAX]]
// CHECK: tt.store {{.*}}, [[IDX]] {{.*}}:
// CHECK-COUNT-2: arith.subf {{.*}}: tensor<64x256xf32, #mma>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @col_argmax_feeds_partitioned_tile(%desc_a: !tt.tensordesc<128x64xf16>, %desc_b: !tt.tensordesc<64x256xf16>, %out: !tt.ptr<f32>, %out_idx: !tt.ptr<i32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %range = tt.make_range {async_task_id = array<i32: 1, 2>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %range2d = tt.expand_dims %range {async_task_id = array<i32: 1, 2>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %rows = tt.broadcast %range2d {async_task_id = array<i32: 1, 2>} : tensor<128x1xi32, #mma> -> tensor<128x256xi32, #mma>
    %max:2 = "tt.reduce"(%dot, %rows) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lv: f32, %li: i32, %rv: f32, %ri: i32):
      %gt = arith.cmpf ogt, %lv, %rv : f32
      %v = arith.select %gt, %lv, %rv : f32
      %i = arith.select %gt, %li, %ri : i32
      tt.reduce.return %v, %i : f32, i32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<128x256xf32, #mma>, tensor<128x256xi32, #mma>) -> (tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>)
    %max2d = tt.expand_dims %max#0 {async_task_id = array<i32: 1, 2>, axis = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xf32, #mma>
    %maxb = tt.broadcast %max2d {async_task_id = array<i32: 1, 2>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
    %centered = arith.subf %dot, %maxb {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma>
    %tile_ptr = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #mma>
    tt.store %tile_ptr, %centered {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f32>, #mma>
    %idx_ptr = tt.splat %out_idx {async_task_id = array<i32: 1, 2>} : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.store %idx_ptr, %max#1 {async_task_id = array<i32: 1, 2>} : tensor<256x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.return
  }
}

// -----

// A reduction with a defined ordering cannot be split into per-partition
// partials without changing its reduction tree, so data partitioning is
// skipped and the dot keeps its full shape.

// CHECK-LABEL: @ordered_col_reduce_skips_partitioning
// CHECK: ttng.warp_group_dot {{.*}} -> tensor<128x256xf32, #mma>
// CHECK: "tt.reduce"
// CHECK: (tensor<128x256xf32, #mma>) -> tensor<256xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ordered_col_reduce_skips_partitioning(%desc_a: !tt.tensordesc<128x64xf16>, %desc_b: !tt.tensordesc<64x256xf16>, %out: !tt.ptr<f32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    // expected-remark @below {{skipping data partitioning: combining per-partition results of a reduction over the partitioned dim would change its reduction_ordering}}
    %sum = "tt.reduce"(%dot) <{axis = 0 : i32, reduction_ordering = "inner_tree"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<128x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %ptr = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.store %ptr, %sum {async_task_id = array<i32: 1, 2>} : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.return
  }
}

// -----

// After a transpose the partitioned M dim is dim 1. Reducing axis 1 (a column
// sum over M) combines partials; reducing axis 0 (a row sum over N) drops a
// dim before the partitioned one, so its result stays partitioned along dim 0.

// CHECK-LABEL: @trans_row_and_col_reduce
// CHECK: [[T0:%.*]] = tt.trans {{.*}} -> tensor<256x64xf32
// CHECK: [[R0:%.*]] = "tt.reduce"([[T0]]) <{axis = 0 : i32
// CHECK: -> tensor<64xf32
// CHECK: [[C0:%.*]] = "tt.reduce"([[T0]]) <{axis = 1 : i32
// CHECK: -> tensor<256xf32
// CHECK: tt.store {{.*}}, [[R0]] {{.*}}: tensor<64x!tt.ptr<f32>
// CHECK: [[T1:%.*]] = tt.trans {{.*}} -> tensor<256x64xf32
// CHECK: [[R1:%.*]] = "tt.reduce"([[T1]]) <{axis = 0 : i32
// CHECK: [[C1:%.*]] = "tt.reduce"([[T1]]) <{axis = 1 : i32
// CHECK: [[SUM:%.*]] = arith.addf [[C0]], [[C1]] {{.*}}: tensor<256xf32
// CHECK: tt.store {{.*}}, [[SUM]] {{.*}}: tensor<256x!tt.ptr<f32>
// CHECK: tt.store {{.*}}, [[R1]] {{.*}}: tensor<64x!tt.ptr<f32>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#tr = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @trans_row_and_col_reduce(%desc_a: !tt.tensordesc<128x64xf16>, %desc_b: !tt.tensordesc<64x256xf16>, %out_row: !tt.ptr<f32>, %out_col: !tt.ptr<f32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %cvt = ttg.convert_layout %dot {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>
    %t = tt.trans %cvt {async_task_id = array<i32: 1, 2>, order = array<i32: 1, 0>} : tensor<128x256xf32, #blocked> -> tensor<256x128xf32, #tr>
    %row = "tt.reduce"(%t) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<256x128xf32, #tr>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #tr}>>
    %col = "tt.reduce"(%t) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<256x128xf32, #tr>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #tr}>>
    %row_ptr = tt.splat %out_row {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #tr}>>
    tt.store %row_ptr, %row {async_task_id = array<i32: 1, 2>} : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #tr}>>
    %col_ptr = tt.splat %out_col {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #tr}>>
    tt.store %col_ptr, %col {async_task_id = array<i32: 1, 2>} : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #tr}>>
    tt.return
  }
}
