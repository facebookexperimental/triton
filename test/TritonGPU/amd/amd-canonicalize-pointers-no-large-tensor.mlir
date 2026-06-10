// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=false" -canonicalize -verify-diagnostics | FileCheck %s

// this case is copied from amd-canonicalize-pointers-no-large-tensor.mlir. With
// enable-large-tensor-ptr-canon=false, the input is not changed at all.
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @conversion1
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.load %4 : tensor<1024x!tt.ptr<f32>>
    tt.return %5 : tensor<1024xf32>
  }
}

// CHECK: %[[ADDPTR:.*]] = tt.addptr
// CHECK:                = tt.load %[[ADDPTR]]

// -----
// Verify that scf.if with mixed promotable/non-promotable pointer yields works.
// One branch yields a fat ptr (base, offset) and the other yields a single ptr.
// The IfOp conversion must reconcile them by materializing the fat ptr back
// with addptr.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: _if_select_ptr
  tt.func public @_if_select_ptr(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c9_i32 = arith.constant 9 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi sge, %0, %c9_i32 : i32
    %2 = arith.muli %0, %arg3 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<bf16>, i32
    %4 = arith.muli %0, %arg4 : i32
    %5 = tt.addptr %arg1, %4 : !tt.ptr<bf16>, i32
    %6 = scf.if %1 -> (!tt.ptr<bf16>) {
      scf.yield %3 : !tt.ptr<bf16>
    } else {
      scf.yield %5 : !tt.ptr<bf16>
    }
    %7 = tt.load %6 : !tt.ptr<bf16>
    tt.store %arg2, %7 : !tt.ptr<bf16>
    tt.return
  }
}

// The scf.if should survive with addptr materialized inside the then branch.
// CHECK: scf.if
// CHECK:   tt.addptr
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   scf.yield
// CHECK: }
// CHECK: tt.load

// -----
// Verify that a scalar select no longer crashes
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: _scalar_select
  tt.func public @_scalar_select(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c9_i32 = arith.constant 9 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.addptr %arg3, %0 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = arith.addi %1, %4 : i32
    %6 = arith.addi %0, %c1_i32 : i32
    %7 = tt.addptr %arg3, %6 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = arith.cmpi sge, %2, %c9_i32 : i32
    %10 = tt.addptr %arg0, %5 : !tt.ptr<bf16>, i32
    %11 = arith.muli %5, %arg8 : i32
    %12 = arith.muli %2, %arg9 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = tt.addptr %arg1, %13 : !tt.ptr<bf16>, i32
    %15 = tt.addptr %arg4, %0 : !tt.ptr<i32>, i32
    %16 = tt.load %15 : !tt.ptr<i32>
    %17 = tt.addptr %arg5, %0 : !tt.ptr<i32>, i32
    %18 = tt.load %17 : !tt.ptr<i32>
    %19 = arith.addi %16, %18 : i32
    %20 = arith.subi %8, %5 : i32
    %21 = arith.subi %19, %20 : i32
    %22 = arith.subi %2, %c9_i32 : i32
    %23 = arith.muli %22, %arg7 : i32
    %24 = arith.muli %21, %arg6 : i32
    %25 = arith.addi %23, %24 : i32
    %26 = tt.addptr %arg2, %25 : !tt.ptr<bf16>, i32
    // CHECK-COUNT-2: tt.addptr
    // CHECK: arith.select
    %27 = arith.select %9, %26, %14 : !tt.ptr<bf16>
    %28 = tt.load %10 : !tt.ptr<bf16>
    tt.store %27, %28 : !tt.ptr<bf16>
    tt.return
  }
}

// -----
// Verify that nested scf.if with mixed promotable/non-promotable pointers
// across multiple levels doesn't crash. The inner scf.if ops have yields
// in opsToRewrite (traced from a tracked arg) but no fat pointer offsets
// because arith.select collapsed the fat ptr. Without the isLegal fix,
// the inner scf.if ops are incorrectly marked illegal and fail to legalize.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: _nested_if_select_ptr
  tt.func public @_nested_if_select_ptr(
      %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg4: i32 {tt.divisibility = 16 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c9_i32 = arith.constant 9 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi sge, %0, %c9_i32 : i32
    %2 = arith.cmpi sge, %0, %c5_i32 : i32
    %3 = arith.muli %0, %arg4 : i32
    %4 = tt.addptr %arg2, %3 : !tt.ptr<bf16>, i32
    // Outer scf.if: then yields tracked ptr, else yields result of nested scf.if
    %5:2 = scf.if %1 -> (!tt.ptr<bf16>, i32) {
      scf.yield %4, %arg5 : !tt.ptr<bf16>, i32
    } else {
      // Inner scf.if: both branches yield untracked/collapsed ptrs
      %inner = scf.if %2 -> (!tt.ptr<bf16>) {
        scf.yield %arg0 : !tt.ptr<bf16>
      } else {
        %sel = arith.select %1, %arg1, %arg3 : !tt.ptr<bf16>
        scf.yield %sel : !tt.ptr<bf16>
      }
      scf.yield %inner, %arg5 : !tt.ptr<bf16>, i32
    }
    %6 = tt.load %5#0 : !tt.ptr<bf16>
    tt.store %arg3, %6 : !tt.ptr<bf16>
    tt.return
  }
}

// The pass should complete without crashing. The outer scf.if reconciles
// the then branch's fat ptr by materializing addptr. The inner scf.if
// is folded by canonicalization into arith.select.
// CHECK: scf.if
// CHECK:   tt.addptr
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   arith.select
// CHECK:   scf.yield
// CHECK: }
// CHECK: tt.load

// -----
// Regression test for a mixed promotable/non-promotable scf.if that yields a
// *tensor* of pointers (concat/split-style kernels). The non-promotable branch
// (no tt.pointer_range, treated as a large 64-bit pointer) yields a full tensor
// of pointers, while the promotable branch (tt.pointer_range = 32) is
// decomposed into a fat pointer. Reconciling the two must splat the scalar base
// before the addptr; previously the pass emitted an invalid scalar-result
// tt.addptr with a non-scalar (tensor) offset operand and crashed the verifier.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: split_2D_jagged
  tt.func public @split_2D_jagged(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.addptr %arg2, %0 : !tt.ptr<i64>, i32
    %3 = tt.load %2 : !tt.ptr<i64>
    %4 = tt.addptr %2, %c1_i32 : !tt.ptr<i64>, i32
    %5 = tt.load %4 : !tt.ptr<i64>
    %6 = arith.subi %5, %3 : i64
    %7 = tt.addptr %arg3, %0 : !tt.ptr<i64>, i32
    %8 = tt.load %7 : !tt.ptr<i64>
    %9 = tt.addptr %7, %c1_i32 : !tt.ptr<i64>, i32
    %10 = tt.load %9 : !tt.ptr<i64>
    %11 = arith.subi %10, %8 : i64
    %12 = arith.addi %6, %11 : i64
    %13 = arith.extsi %1 : i32 to i64
    %15 = arith.addi %3, %8 : i64
    %16 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %17 = arith.addi %15, %13 : i64
    %18 = arith.extsi %arg7 : i32 to i64
    %19 = arith.muli %17, %18 : i64
    %20 = tt.addptr %arg0, %19 : !tt.ptr<f32>, i64
    %21 = tt.splat %20 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %22 = tt.addptr %21, %16 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    %23 = arith.cmpi slt, %13, %6 : i64
    %24 = arith.cmpi sge, %1, %c0_i32 : i32
    %25 = arith.andi %23, %24 : i1
    %26 = scf.if %25 -> (tensor<512x!tt.ptr<f32>, #blocked>) {
      %30 = arith.addi %13, %3 : i64
      %31 = arith.extsi %arg8 : i32 to i64
      %32 = arith.muli %30, %31 : i64
      %33 = tt.addptr %arg4, %32 : !tt.ptr<f32>, i64
      %34 = tt.splat %33 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
      %35 = tt.addptr %34, %16 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
      scf.yield %35 : tensor<512x!tt.ptr<f32>, #blocked>
    } else {
      %30 = arith.subi %13, %6 : i64
      %31 = arith.cmpi slt, %1, %c0_i32 : i32
      %32 = arith.select %31, %13, %30 : i64
      %33 = arith.addi %32, %8 : i64
      %34 = arith.extsi %arg9 : i32 to i64
      %35 = arith.muli %33, %34 : i64
      %36 = tt.addptr %arg5, %35 : !tt.ptr<f32>, i64
      %37 = tt.splat %36 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
      %38 = tt.addptr %37, %16 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
      scf.yield %38 : tensor<512x!tt.ptr<f32>, #blocked>
    }
    %27 = tt.splat %arg6 : i32 -> tensor<512xi32, #blocked>
    %28 = arith.cmpi slt, %16, %27 : tensor<512xi32, #blocked>
    %29 = tt.load %22, %28 : tensor<512x!tt.ptr<f32>, #blocked>
    tt.store %26, %29, %28 : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// The pass should complete without crashing. The scf.if yields a tensor of
// pointers; the promotable (else) branch materializes its fat pointer back into
// a tensor pointer with splat + addptr.
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (tensor<512x!tt.ptr<f32>, #blocked>)
// CHECK:   tt.splat
// CHECK:   tt.addptr {{.*}} : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
// CHECK:   scf.yield {{.*}} : tensor<512x!tt.ptr<f32>, #blocked>
// CHECK: } else {
// CHECK:   tt.splat
// CHECK:   tt.addptr {{.*}} : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
// CHECK:   scf.yield {{.*}} : tensor<512x!tt.ptr<f32>, #blocked>
// CHECK: }
// CHECK: tt.store %[[IF]]
