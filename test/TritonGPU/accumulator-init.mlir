// RUN: triton-opt %s -split-input-file -tritongpu-optimize-accumulator-init | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @constant_init
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %[[FALSE]]
  tt.func @constant_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %cst_2 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @constant_init_integer
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %[[FALSE]]
  tt.func @constant_init_integer(%A: !ttg.memdesc<128x64xi8, #shared, #smem>, %B: !ttg.memdesc<64x16xi8, #shared1, #smem>, %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xi32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0> : tensor<128x16xi32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xi32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %cst_2 : !ttg.memdesc<128x64xi8, #shared, #smem> * !ttg.memdesc<64x16xi8, #shared1, #smem> -> tensor<128x16xi32, #mma1>
      scf.yield %acc: tensor<128x16xi32, #mma1>
    }
    tt.return %17 : tensor<128x16xi32, #mma1>
  }

// CHECK-LABEL: @if_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @if_after_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_after_mma_invert
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[CND]]
  tt.func @if_after_mma_invert(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_before_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      }
      %acc = ttng.warp_group_dot %A, %B, %acc_ : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_before_mma_invert
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[USE_ACC]], %[[FALSE]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_mma_invert(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      }
      %acc = ttng.warp_group_dot %A, %B, %acc_ : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @sel_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @sel_after_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @sel_before_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @sel_before_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = arith.select %cnd, %cst_2, %arg4 : tensor<128x16xf32, #mma1>
      %acc = ttng.warp_group_dot %A, %B, %acc_ : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }


// Check that we look only at the zeroing directly preceding the mma

// CHECK-LABEL: @if_before_and_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[C0_TENSOR]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_and_after_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_0 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      }
      %acc = ttng.warp_group_dot %A, %B, %acc_0 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_1 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_1: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @two_ifs_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[C0_TENSOR]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_CND]]
// CHECK: else
// CHECK: scf.yield %[[ACC_CND]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @two_ifs_after_mma(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_0 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      %acc_1 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc_0 : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_1: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

  // CHECK-LABEL: @zero_init_dist_2
  tt.func @zero_init_dist_2(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    // CHECK: scf.for {{.*}} = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg{{[1-9]+}} = %{{.*}}, %[[ACC:.*]] = %[[CST]], %[[INIT_FLAG:.*]] = %false)
    %17:2 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %cst_2) -> (tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      // CHECK: %2 = ttng.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[INIT_FLAG]]
      %acc = ttng.warp_group_dot %A, %B, %arg5 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      // CHECK: scf.yield {{.*}}, {{.*}}, %true
      scf.yield %acc_, %arg4: tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @packed_mul_zero_init
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = {{.*}}, %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: tt.elementwise_inline_asm {{.*}} %[[ACC]], {{.*}}
// CHECK: ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %[[USE_ACC]]
  tt.func @packed_mul_zero_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %scale: tensor<128x16xf32, #mma1>) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %result = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x16xf32, #mma1>) : i32 {
      %scaled = tt.elementwise_inline_asm "mul.f32x2 $0, $2, $4;" {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %acc, %scale : tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %next = ttng.warp_group_dot %A, %B, %scaled : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %next : tensor<128x16xf32, #mma1>
    }
    tt.return %result : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @select_preserves_packed_mul_zero_init
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = {{.*}}, %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[SCALED:.+]] = tt.elementwise_inline_asm {{.*}} %[[ACC]], {{.*}}
// CHECK: %[[SELECTED:.+]] = arith.select {{.*}}, %[[SCALED]], %[[ACC]]
// CHECK: ttng.warp_group_dot {{.*}}, {{.*}}, %[[SELECTED]], %[[USE_ACC]]
  tt.func @select_preserves_packed_mul_zero_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %scale: tensor<128x16xf32, #mma1>, %condition: tensor<128x16xi1, #mma1>) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %result = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x16xf32, #mma1>) : i32 {
      %scaled = tt.elementwise_inline_asm "mul.f32x2 $0, $2, $4;" {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %acc, %scale : tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %selected = arith.select %condition, %scaled, %acc : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1>
      %next = ttng.warp_group_dot %A, %B, %selected : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %next : tensor<128x16xf32, #mma1>
    }
    tt.return %result : tensor<128x16xf32, #mma1>
  }

// The packed mul as the kernels actually emit it: movs around a single
// mul.f32x2 (see inline_ptx_lib.py _mul_f32x2). Zero still propagates, so the
// init must fold -- a matcher insisting on one instruction would reject every
// real producer and silently disable this optimization.
// CHECK-LABEL: @packed_mul_block_zero_init
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = {{.*}}, %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %[[USE_ACC]]
  tt.func @packed_mul_block_zero_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %scale: tensor<128x16xf32, #mma1>) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %result = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x16xf32, #mma1>) : i32 {
      %scaled = tt.elementwise_inline_asm "{\0A  .reg .b64 ra, rb, rc;\0A  mov.b64 ra, { $2, $3 };\0A  mov.b64 rb, { $4, $5 };\0A  mul.f32x2 rc, ra, rb;\0A  mov.b64 { $0, $1 }, rc;\0A}" {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %acc, %scale : tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %next = ttng.warp_group_dot %A, %B, %scaled : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %next : tensor<128x16xf32, #mma1>
    }
    tt.return %result : tensor<128x16xf32, #mma1>
  }

// This asm multiplies and THEN adds, so 0 * scale + 1.0 = 1.0 and the zero is
// NOT preserved. It still contains the mul.f32x2 token and satisfies every
// structural check, so a substring match would wrongly fold the init and
// corrupt the first iteration. The accumulator must stay explicit: no
// use-accumulator flag is threaded through the loop.
// CHECK-LABEL: @packed_mul_then_add_no_zero_init
// CHECK-NOT: arith.constant false
// CHECK: scf.for {{.*}} iter_args({{.*}}) ->
// CHECK: ttng.warp_group_dot
  tt.func @packed_mul_then_add_no_zero_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %scale: tensor<128x16xf32, #mma1>) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %result = scf.for %i = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x16xf32, #mma1>) : i32 {
      %scaled = tt.elementwise_inline_asm "mul.f32x2 $0, $2, $4;\0Aadd.f32x2 $0, $0, $0;" {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %acc, %scale : tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %next = ttng.warp_group_dot %A, %B, %scaled : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %next : tensor<128x16xf32, #mma1>
    }
    tt.return %result : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_defines_alternative
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %arg{{.*}} : !ttg.memdesc
  tt.func @if_defines_alternative(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        %acc_alt = arith.addf %acc, %cst_3 : tensor<128x16xf32, #mma1>
        scf.yield %acc_alt : tensor<128x16xf32, #mma1>
      }
      // CHECK: scf.yield {{.*}}, %true
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @non_cond_override
// CHECK: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %arg{{.*}} : !ttg.memdesc
  tt.func @non_cond_override(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.addf %acc, %cst_3 : tensor<128x16xf32, #mma1>
      // CHECK: scf.yield {{.*}}, %true
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }


// Check that we bail out in unsupported cases

// CHECK-LABEL: @non_zero_init
// CHECK-NOT: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !ttg.memdesc
  tt.func @non_zero_init(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = ttng.warp_group_dot %A, %B, %arg4 : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// If the condition is a tensor skip the optimization.
// CHECK-LABEL: @negative_sel_tensor
// CHECK-NOT: %[[ACC_NEXT:.+]] = ttng.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !ttg.memdesc
  tt.func @negative_sel_tensor(%A: !ttg.memdesc<128x64xf16, #shared, #smem>, %B: !ttg.memdesc<64x16xf16, #shared1, #smem>, %cnd: tensor<128x16xi1, #mma1>) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %acc_ = arith.select %cnd, %cst_2, %arg4 : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1>
      %acc = ttng.warp_group_dot %A, %B, %acc_ : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x16xf16, #shared1, #smem> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }
}
