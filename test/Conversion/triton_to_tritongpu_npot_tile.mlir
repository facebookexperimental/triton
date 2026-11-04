// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-remove-layout-conversions -allocate-shared-memory-nv='compute-capability=90' -convert-triton-gpu-to-llvm='compute-capability=90' | FileCheck %s --check-prefix=LLVM

// The tile takes the [96, 64] Product layout and each thread owns 48 elements,
// where padding to [128, 64] would give it 64.
// CHECK: #linear = #ttg.linear<{{.*}}outDimSizes = [96, 64]}>
// CHECK-LABEL: @tile_96x64
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.expand_dims {{.*}} -> tensor<96x1xi32, #linear>
// CHECK: tt.expand_dims {{.*}} -> tensor<1x64xi32, #linear>
// CHECK: tt.broadcast {{.*}} -> tensor<96x64xi32, #linear>
// CHECK: tt.load {{.*}} : tensor<96x64x!tt.ptr<f32>, #linear>
// CHECK: tt.store {{.*}} : tensor<96x64x!tt.ptr<f32>, #linear>
// CHECK-NOT: ttg.convert_layout
// LLVM-LABEL: llvm.func @tile_96x64
// LLVM-COUNT-48: llvm.load {{.*}} : !llvm.ptr<1> -> f32
// LLVM-NOT: llvm.load {{.*}} !llvm.ptr<1>
// LLVM-COUNT-48: llvm.store {{.*}} : f32, !llvm.ptr<1>
// LLVM-NOT: llvm.store {{.*}} !llvm.ptr<1>
tt.func @tile_96x64(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %m = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<96xi32> -> tensor<96x1xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %stride = arith.constant dense<64> : tensor<96x1xi32>
  %row = arith.muli %m1, %stride : tensor<96x1xi32>
  %rows = tt.broadcast %row : tensor<96x1xi32> -> tensor<96x64xi32>
  %cols = tt.broadcast %n1 : tensor<1x64xi32> -> tensor<96x64xi32>
  %off = arith.addi %rows, %cols : tensor<96x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %p = tt.addptr %sp, %off : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %v = tt.load %p : tensor<96x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %q = tt.addptr %dp, %off : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  tt.store %q, %v : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK: #linear = #ttg.linear<{{.*}}outDimSizes = [64, 96]}>
// CHECK-LABEL: @tile_64x96
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.expand_dims {{.*}} -> tensor<64x1xi32, #linear>
// CHECK: tt.expand_dims {{.*}} -> tensor<1x96xi32, #linear>
// CHECK: tt.broadcast {{.*}} -> tensor<64x96xi32, #linear>
// CHECK: tt.load {{.*}} : tensor<64x96x!tt.ptr<f32>, #linear>
// CHECK: tt.store {{.*}} : tensor<64x96x!tt.ptr<f32>, #linear>
// CHECK-NOT: ttg.convert_layout
// LLVM-LABEL: llvm.func @tile_64x96
// LLVM-COUNT-48: llvm.load {{.*}} : !llvm.ptr<1> -> f32
// LLVM-NOT: llvm.load {{.*}} !llvm.ptr<1>
// LLVM-COUNT-48: llvm.store {{.*}} : f32, !llvm.ptr<1>
// LLVM-NOT: llvm.store {{.*}} !llvm.ptr<1>
tt.func @tile_64x96(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %n = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<96xi32> -> tensor<1x96xi32>
  %stride = arith.constant dense<96> : tensor<64x1xi32>
  %row = arith.muli %m1, %stride : tensor<64x1xi32>
  %rows = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x96xi32>
  %cols = tt.broadcast %n1 : tensor<1x96xi32> -> tensor<64x96xi32>
  %off = arith.addi %rows, %cols : tensor<64x96xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>>
  %p = tt.addptr %sp, %off : tensor<64x96x!tt.ptr<f32>>, tensor<64x96xi32>
  %v = tt.load %p : tensor<64x96x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>>
  %q = tt.addptr %dp, %off : tensor<64x96x!tt.ptr<f32>>, tensor<64x96xi32>
  tt.store %q, %v : tensor<64x96x!tt.ptr<f32>>
  tt.return
}

// -----

// Power-of-two control.
// CHECK-LABEL: @tile_64x64
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.expand_dims {{.*}} -> tensor<64x1xi32, #blocked>
// CHECK: tt.expand_dims {{.*}} -> tensor<1x64xi32, #blocked>
// CHECK: tt.broadcast {{.*}} -> tensor<64x64xi32, #blocked>
// CHECK: tt.load {{.*}} : tensor<64x64x!tt.ptr<f32>, #blocked>
// CHECK: tt.store {{.*}} : tensor<64x64x!tt.ptr<f32>, #blocked>
// CHECK-NOT: ttg.convert_layout
// LLVM-LABEL: llvm.func @tile_64x64
// LLVM-COUNT-32: llvm.load {{.*}} : !llvm.ptr<1> -> f32
// LLVM-NOT: llvm.load {{.*}} !llvm.ptr<1>
// LLVM-COUNT-32: llvm.store {{.*}} : f32, !llvm.ptr<1>
// LLVM-NOT: llvm.store {{.*}} !llvm.ptr<1>
tt.func @tile_64x64(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %stride = arith.constant dense<64> : tensor<64x1xi32>
  %row = arith.muli %m1, %stride : tensor<64x1xi32>
  %rows = tt.broadcast %row : tensor<64x1xi32> -> tensor<64x64xi32>
  %cols = tt.broadcast %n1 : tensor<1x64xi32> -> tensor<64x64xi32>
  %off = arith.addi %rows, %cols : tensor<64x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
  %p = tt.addptr %sp, %off : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
  %v = tt.load %p : tensor<64x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
  %q = tt.addptr %dp, %off : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
  tt.store %q, %v : tensor<64x64x!tt.ptr<f32>>
  tt.return
}
