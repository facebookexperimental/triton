// RUN: triton-opt -split-input-file %s --tritongpu-print-tlx | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add2_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<1024xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK:      @triton.jit
// CHECK-NEXT: def add2_kernel(v_0, v_1, v_2, v_3):
// CHECK-NEXT:   v_4 = 1024
// CHECK-NEXT:   v_5 = tl.program_id(0)
// CHECK-NEXT:   v_6 = v_5 * v_4
// CHECK-NEXT:   v_7 = tl.arange(0, 1024)
// CHECK-NEXT:   v_8 = tl.full((1024, ), v_6, tl.int32)
// CHECK-NEXT:   v_9 = v_8 + v_7
// CHECK-NEXT:   v_10 = tl.full((1024, ), v_3, tl.int32)
// CHECK-NEXT:   v_11 = v_9 < v_10
// CHECK-NEXT:   v_12 = tl.full((1024, ), v_0, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_13 = v_12 + v_9
// CHECK-NEXT:   v_14 = tl.load(v_13, v_11)
// CHECK-NEXT:   v_15 = tl.full((1024, ), v_1, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_16 = v_15 + v_9
// CHECK-NEXT:   v_17 = tl.load(v_16, v_11)
// CHECK-NEXT:   v_18 = v_14 + v_17
// CHECK-NEXT:   v_19 = tl.full((1024, ), v_2, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_20 = v_19 + v_9
// CHECK-NEXT:   tl.store(v_20, v_18, v_11)
// CHECK-NEXT:   return

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add2_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %13 = tt.reshape %9 allow_reorder : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked>
    %14 = "tt.reduce"(%13) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %19 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %19 : f32
    }) : (tensor<1024xf32, #blocked>) -> f32
    %15 = tt.reshape %12 allow_reorder : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked>
    %16 = "tt.reduce"(%15) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %19 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %19 : f32
    }) : (tensor<1024xf32, #blocked>) -> f32
    %17 = arith.addf %14, %16 : f32
    %18 = tt.addptr %arg2, %0 : !tt.ptr<f32>, i32
    tt.store %18, %17 : !tt.ptr<f32>
    tt.return
  }
}