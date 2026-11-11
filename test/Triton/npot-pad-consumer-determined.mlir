// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=4' -verify-diagnostics | FileCheck %s

// The matmul index idiom: the N-248 index values take the B/C tiles' layout, so
// tiles alone decide the connected component and it stays exact at 4 warps.
// CHECK-LABEL: @matmul_index_248
module {
  tt.func @matmul_index_248(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32, %K: i32 {tt.divisibility = 16 : i32}, %sam: i32 {tt.divisibility = 16 : i32}, %sbk: i32, %scm: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x248xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf16>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<16> : tensor<128x16xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x248xf32>
    %c16_i32 = arith.constant 16 : i32
    %c248_i32 = arith.constant 248 : i32
    %c128_i32 = arith.constant 128 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c128_i32 : i32
    %offs_m_3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<128xi32>
    %offs_m_5 = arith.addi %offs_m_4, %offs_m_3 : tensor<128xi32>
    %offs_n = arith.muli %pid_n, %c248_i32 : i32
  // CHECK: tt.make_range {end = 248 : i32, start = 0 : i32} : tensor<248xi32>
    %offs_n_6 = tt.make_range {end = 248 : i32, start = 0 : i32} : tensor<248xi32>
    %offs_n_7 = tt.splat %offs_n : i32 -> tensor<248xi32>
    %offs_n_8 = arith.addi %offs_n_7, %offs_n_6 : tensor<248xi32>
    %offs_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %a_ptrs = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %a_ptrs_9 = tt.splat %sam : i32 -> tensor<128x1xi32>
    %a_ptrs_10 = arith.muli %a_ptrs, %a_ptrs_9 : tensor<128x1xi32>
    %a_ptrs_11 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %a_ptrs_12 = tt.addptr %a_ptrs_11, %a_ptrs_10 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    %a_ptrs_13 = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %a_ptrs_14 = tt.broadcast %a_ptrs_12 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x16x!tt.ptr<f16>>
    %a_ptrs_15 = tt.broadcast %a_ptrs_13 : tensor<1x16xi32> -> tensor<128x16xi32>
    %a_ptrs_16 = tt.addptr %a_ptrs_14, %a_ptrs_15 : tensor<128x16x!tt.ptr<f16>>, tensor<128x16xi32>
    %b_ptrs = tt.expand_dims %offs_k {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %b_ptrs_17 = tt.splat %sbk : i32 -> tensor<16x1xi32>
    %b_ptrs_18 = arith.muli %b_ptrs, %b_ptrs_17 : tensor<16x1xi32>
    %b_ptrs_19 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
    %b_ptrs_20 = tt.addptr %b_ptrs_19, %b_ptrs_18 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
  // CHECK: tt.expand_dims %{{.*}} : tensor<248xi32> -> tensor<1x248xi32>
    %b_ptrs_21 = tt.expand_dims %offs_n_8 {axis = 0 : i32} : tensor<248xi32> -> tensor<1x248xi32>
    %b_ptrs_22 = tt.broadcast %b_ptrs_20 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x248x!tt.ptr<f16>>
    %b_ptrs_23 = tt.broadcast %b_ptrs_21 : tensor<1x248xi32> -> tensor<16x248xi32>
    %b_ptrs_24 = tt.addptr %b_ptrs_22, %b_ptrs_23 : tensor<16x248x!tt.ptr<f16>>, tensor<16x248xi32>
    %a = tt.splat %M : i32 -> tensor<128x1xi32>
    %a_25 = arith.cmpi slt, %a_ptrs, %a : tensor<128x1xi32>
    %a_26 = tt.splat %K : i32 -> tensor<1x16xi32>
    %a_27 = tt.broadcast %a_25 : tensor<128x1xi1> -> tensor<128x16xi1>
    %b = tt.splat %K : i32 -> tensor<16x1xi32>
    %b_28 = tt.splat %N : i32 -> tensor<1x248xi32>
    %b_29 = arith.cmpi slt, %b_ptrs_21, %b_28 : tensor<1x248xi32>
    %b_30 = tt.broadcast %b_29 : tensor<1x248xi1> -> tensor<16x248xi1>
    %b_ptrs_31 = arith.muli %sbk, %c16_i32 : i32
    %b_ptrs_32 = tt.splat %b_ptrs_31 : i32 -> tensor<16x248xi32>
    %acc:3 = scf.for %kk = %c0_i32 to %K step %c16_i32 iter_args(%a_ptrs_39 = %a_ptrs_16, %b_ptrs_40 = %b_ptrs_24, %acc_41 = %cst_2) -> (tensor<128x16x!tt.ptr<f16>>, tensor<16x248x!tt.ptr<f16>>, tensor<128x248xf32>)  : i32 {
      %a_42 = tt.splat %kk : i32 -> tensor<1x16xi32>
      %a_43 = arith.addi %a_ptrs_13, %a_42 : tensor<1x16xi32>
      %a_44 = arith.cmpi slt, %a_43, %a_26 : tensor<1x16xi32>
      %a_45 = tt.broadcast %a_44 : tensor<1x16xi1> -> tensor<128x16xi1>
      %a_46 = arith.andi %a_27, %a_45 : tensor<128x16xi1>
      %a_47 = tt.load %a_ptrs_39, %a_46, %cst_0 : tensor<128x16x!tt.ptr<f16>>
      %b_48 = tt.splat %kk : i32 -> tensor<16x1xi32>
      %b_49 = arith.addi %b_ptrs, %b_48 : tensor<16x1xi32>
      %b_50 = arith.cmpi slt, %b_49, %b : tensor<16x1xi32>
      %b_51 = tt.broadcast %b_50 : tensor<16x1xi1> -> tensor<16x248xi1>
      %b_52 = arith.andi %b_51, %b_30 : tensor<16x248xi1>
  // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x248x!tt.ptr<f16>>
      %b_53 = tt.load %b_ptrs_40, %b_52, %cst : tensor<16x248x!tt.ptr<f16>>
      tt.store %b_ptrs_40, %b_53, %b_52 : tensor<16x248x!tt.ptr<f16>>
      %acc_54 = arith.addf %acc_41, %acc_41 : tensor<128x248xf32>
      %a_ptrs_55 = tt.addptr %a_ptrs_39, %cst_1 : tensor<128x16x!tt.ptr<f16>>, tensor<128x16xi32>
      %b_ptrs_56 = tt.addptr %b_ptrs_40, %b_ptrs_32 : tensor<16x248x!tt.ptr<f16>>, tensor<16x248xi32>
      scf.yield %a_ptrs_55, %b_ptrs_56, %acc_54 : tensor<128x16x!tt.ptr<f16>>, tensor<16x248x!tt.ptr<f16>>, tensor<128x248xf32>
    }
    %c_ptrs = tt.splat %scm : i32 -> tensor<128x1xi32>
    %c_ptrs_33 = arith.muli %a_ptrs, %c_ptrs : tensor<128x1xi32>
    %c_ptrs_34 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %c_ptrs_35 = tt.addptr %c_ptrs_34, %c_ptrs_33 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    %c_ptrs_36 = tt.broadcast %c_ptrs_35 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x248x!tt.ptr<f16>>
    %c_ptrs_37 = tt.broadcast %b_ptrs_21 : tensor<1x248xi32> -> tensor<128x248xi32>
    %c_ptrs_38 = tt.addptr %c_ptrs_36, %c_ptrs_37 : tensor<128x248x!tt.ptr<f16>>, tensor<128x248xi32>
    %0 = tt.splat %M : i32 -> tensor<128x1xi32>
    %1 = arith.cmpi slt, %a_ptrs, %0 : tensor<128x1xi32>
    %2 = tt.splat %N : i32 -> tensor<1x248xi32>
    %3 = arith.cmpi slt, %b_ptrs_21, %2 : tensor<1x248xi32>
    %4 = tt.broadcast %1 : tensor<128x1xi1> -> tensor<128x248xi1>
    %5 = tt.broadcast %3 : tensor<1x248xi1> -> tensor<128x248xi1>
    %6 = arith.andi %4, %5 : tensor<128x248xi1>
    %7 = arith.truncf %acc#2 : tensor<128x248xf32> to tensor<128x248xf16>
  // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x248x!tt.ptr<f16>>
    tt.store %c_ptrs_38, %7, %6 : tensor<128x248x!tt.ptr<f16>>
    tt.return
  }
}
