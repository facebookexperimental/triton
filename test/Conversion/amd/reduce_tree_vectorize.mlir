// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx90a enable-tree-reduction=true' -cse | FileCheck %s --check-prefix=GFX90A
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx942 enable-tree-reduction=true' -cse | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx950 enable-tree-reduction=true' -cse | FileCheck %s --check-prefix=GFX950
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx1250 enable-tree-reduction=true' -cse | FileCheck %s --check-prefix=GFX1250
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm='gfx-arch=gfx942 enable-tree-reduction=false' -cse | FileCheck %s --check-prefix=LINEAR

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: reduce_f16
  // GFX942: llvm.fadd {{.*}} : vector<2xf16>
  // GFX950-LABEL: reduce_f16
  // GFX950: llvm.fadd {{.*}} : vector<2xf16>
  // LINEAR-LABEL: reduce_f16
  // LINEAR-NOT: vector<2xf16>
  // LINEAR: llvm.fadd {{.*}} : f16
  tt.func public @reduce_f16(%arg0: tensor<1x256xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x256xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX90A-LABEL: reduce_f32
  // GFX90A: llvm.fadd {{.*}} : vector<2xf32>
  // GFX942-LABEL: reduce_f32
  // GFX942: llvm.fadd {{.*}} : vector<2xf32>
  // GFX950-LABEL: reduce_f32
  // GFX950: llvm.fadd {{.*}} : vector<2xf32>
  // LINEAR-LABEL: reduce_f32
  // LINEAR-NOT: vector<2xf32>
  // LINEAR: llvm.fadd {{.*}} : f32
  tt.func public @reduce_f32(%arg0: tensor<1x256xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x256xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}

// -----

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: reduce_f16_tree_vectorize
  // GFX1250: llvm.fadd {{.*}} : vector<2xf16>
  tt.func public @reduce_f16_tree_vectorize(%arg0: tensor<1x128xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x128xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_f32_tree_vectorize
  // GFX1250: llvm.fadd {{.*}} : vector<2xf32>
  tt.func public @reduce_f32_tree_vectorize(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // Ternary tree reduction for max/min: generates a chain of 3 dependent ops
  // per group so LLVM can fold into v_maximum3/v_minimum3/v_max3/v_min3.

  // GFX1250-LABEL: reduce_maximum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.maximum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.maximum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_maximum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maximumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_minimum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.minimum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.minimum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.minimum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_minimum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minimumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_maxnum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.maxnum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.maxnum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_maxnum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maxnumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_minnum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.minnum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.minnum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_minnum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minnumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}

// -----

// Register bit 0 is separated from bits 2--4 by a lane bit, forming eight
// two-value register groups along the reduction axis. Since both warp bits
// also move that axis, this exercises the multi-group reduction path.
#multi_group = #ttg.linear<{register = [[1, 0], [4, 0], [8, 0], [16, 0]], lane = [[2, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4]], warp = [[128, 0], [256, 0]], block = []}>
#rows = #ttg.slice<{dim = 0, parent = #multi_group}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Unordered reductions may merge all eight groups before vectorization:
  // 16 register values become eight packed pairs and one seven-node vector
  // tree, followed by a single horizontal scalar add.
  // GFX950-LABEL: reduce_multi_register_groups_unordered
  // GFX950-COUNT-7: llvm.fadd {{.*}} : vector<2xf32>
  // GFX950-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // GFX950: llvm.extractelement
  // GFX950: llvm.extractelement
  // GFX950: llvm.fadd {{.*}} : f32
  tt.func public @reduce_multi_register_groups_unordered(%arg0: tensor<512x8xf32, #multi_group>) {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<512x8xf32, #multi_group>) -> tensor<8xf32, #rows>
    tt.return
  }

  // Explicitly ordered reductions continue to reduce each register group
  // independently and therefore do not use packed vector combines.
  // GFX950-LABEL: reduce_multi_register_groups_inner_tree
  // GFX950-NOT: vector<2xf32>
  // GFX950: llvm.return
  tt.func public @reduce_multi_register_groups_inner_tree(%arg0: tensor<512x8xf32, #multi_group>) {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32, reduction_ordering = "inner_tree"}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<512x8xf32, #multi_group>) -> tensor<8xf32, #rows>
    tt.return
  }
}
