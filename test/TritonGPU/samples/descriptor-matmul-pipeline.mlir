// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.

// CHECK: #[[$ATTR_0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
// CHECK: #[[$ATTR_1:.+]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
// CHECK: #[[$ATTR_2:.+]] = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
// CHECK: #[[$ATTR_3:.+]] = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK: #[[$ATTR_4:.+]] = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
// CHECK: #[[$ATTR_5:.+]] = #ttg.shared_memory
// To regenerate this test case, run `make golden-samples` in the triton root directory
// RUN: triton-opt %s -split-input-file -tritongpu-loop-scheduling -tritongpu-pipeline -canonicalize | FileCheck --dump-input-context=51 %s
// CHECK-LABEL:   tt.func public @matmul_kernel_with_descriptors(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_4:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_5:.*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 127 : i32
// CHECK:           %[[VAL_17:.*]] = arith.constant 255 : i32
// CHECK:           %[[VAL_18:.*]] = arith.constant 63 : i32
// CHECK:           %[[VAL_19:.*]] = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_20:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_3]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_22:.*]] = arith.divsi %[[VAL_21]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_4]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_24:.*]] = arith.divsi %[[VAL_23]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_24]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_26:.*]] = arith.divsi %[[VAL_20]], %[[VAL_25]] : i32
// CHECK:           %[[VAL_27:.*]] = arith.muli %[[VAL_26]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_22]], %[[VAL_27]] : i32
// CHECK:           %[[VAL_29:.*]] = arith.minsi %[[VAL_28]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_30:.*]] = arith.remsi %[[VAL_20]], %[[VAL_29]] : i32
// CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_27]], %[[VAL_30]] : i32
// CHECK:           %[[VAL_32:.*]] = arith.remsi %[[VAL_20]], %[[VAL_25]] : i32
// CHECK:           %[[VAL_33:.*]] = arith.divsi %[[VAL_32]], %[[VAL_29]] : i32
// CHECK:           %[[VAL_34:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK:           %[[VAL_35:.*]] = tt.make_tensor_descriptor %[[VAL_0]], {{\[}}%[[VAL_3]], %[[VAL_5]]], {{\[}}%[[VAL_34]], %[[VAL_14]]] : <f16>, <tensor<128x64xf16>>
// CHECK:           %[[VAL_36:.*]] = tt.make_tensor_descriptor %[[VAL_1]], {{\[}}%[[VAL_4]], %[[VAL_5]]], {{\[}}%[[VAL_34]], %[[VAL_14]]] : <f16>, <tensor<256x64xf16>>
// CHECK:           %[[VAL_37:.*]] = arith.extsi %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_38:.*]] = tt.make_tensor_descriptor %[[VAL_2]], {{\[}}%[[VAL_3]], %[[VAL_4]]], {{\[}}%[[VAL_37]], %[[VAL_14]]] : <f16>, <tensor<128x256xf16>>
// CHECK:           %[[VAL_39:.*]] = arith.muli %[[VAL_31]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_40:.*]] = arith.muli %[[VAL_33]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_41:.*]] = arith.addi %[[VAL_5]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_42:.*]] = arith.divsi %[[VAL_41]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_43:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_44:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_45:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_46:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_12]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.init_barrier %[[VAL_46]], 1 : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_47:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_15]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.init_barrier %[[VAL_47]], 1 : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_48:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_6]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.init_barrier %[[VAL_48]], 1 : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_49:.*]] = arith.cmpi sgt, %[[VAL_42]], %[[VAL_12]] : i32
// CHECK:           %[[VAL_50:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_12]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.barrier_expect %[[VAL_50]], 49152, %[[VAL_49]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_51:.*]] = ttg.memdesc_subview %[[VAL_43]]{{\[}}%[[VAL_12]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:           %[[VAL_52:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_35]] : !tt.tensordesc<tensor<128x64xf16>> to !tt.ptr<i8>
// CHECK:           ttng.async_tma_copy_global_to_local %[[VAL_52]]{{\[}}%[[VAL_39]], %[[VAL_12]]] %[[VAL_51]], %[[VAL_50]], %[[VAL_49]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:           %[[VAL_53:.*]] = ttg.memdesc_subview %[[VAL_44]]{{\[}}%[[VAL_12]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:           %[[VAL_54:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_36]] : !tt.tensordesc<tensor<256x64xf16>> to !tt.ptr<i8>
// CHECK:           ttng.async_tma_copy_global_to_local %[[VAL_54]]{{\[}}%[[VAL_40]], %[[VAL_12]]] %[[VAL_53]], %[[VAL_50]], %[[VAL_49]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:           %[[VAL_55:.*]] = arith.cmpi sgt, %[[VAL_42]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_56:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_15]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.barrier_expect %[[VAL_56]], 49152, %[[VAL_55]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_57:.*]] = ttg.memdesc_subview %[[VAL_43]]{{\[}}%[[VAL_15]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:           %[[VAL_58:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_35]] : !tt.tensordesc<tensor<128x64xf16>> to !tt.ptr<i8>
// CHECK:           ttng.async_tma_copy_global_to_local %[[VAL_58]]{{\[}}%[[VAL_39]], %[[VAL_13]]] %[[VAL_57]], %[[VAL_56]], %[[VAL_55]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:           %[[VAL_59:.*]] = ttg.memdesc_subview %[[VAL_44]]{{\[}}%[[VAL_15]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:           %[[VAL_60:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_36]] : !tt.tensordesc<tensor<256x64xf16>> to !tt.ptr<i8>
// CHECK:           ttng.async_tma_copy_global_to_local %[[VAL_60]]{{\[}}%[[VAL_40]], %[[VAL_13]]] %[[VAL_59]], %[[VAL_56]], %[[VAL_55]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:           %[[VAL_61:.*]]:5 = scf.for %[[VAL_62:.*]] = %[[VAL_12]] to %[[VAL_42]] step %[[VAL_15]] iter_args(%[[VAL_63:.*]] = %[[VAL_19]], %[[VAL_64:.*]] = %[[VAL_13]], %[[VAL_65:.*]] = %[[VAL_15]], %[[VAL_66:.*]] = %[[VAL_8]], %[[VAL_67:.*]] = %[[VAL_12]]) -> (tensor<128x256xf32, #[[$ATTR_1]]>, i32, i32, i32, i32)  : i32 {
// CHECK:             %[[VAL_68:.*]] = arith.subi %[[VAL_42]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_69:.*]] = arith.cmpi slt, %[[VAL_62]], %[[VAL_68]] : i32
// CHECK:             %[[VAL_70:.*]] = arith.addi %[[VAL_66]], %[[VAL_15]] : i32
// CHECK:             %[[VAL_71:.*]] = arith.cmpi slt, %[[VAL_70]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_72:.*]] = arith.select %[[VAL_71]], %[[VAL_70]], %[[VAL_12]] : i32
// CHECK:             %[[VAL_73:.*]] = arith.xori %[[VAL_67]], %[[VAL_15]] : i32
// CHECK:             %[[VAL_74:.*]] = arith.select %[[VAL_71]], %[[VAL_67]], %[[VAL_73]] : i32
// CHECK:             %[[VAL_75:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_72]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:             ttng.wait_barrier %[[VAL_75]], %[[VAL_74]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:             %[[VAL_76:.*]] = ttg.memdesc_subview %[[VAL_44]]{{\[}}%[[VAL_72]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:             %[[VAL_77:.*]] = ttg.memdesc_subview %[[VAL_43]]{{\[}}%[[VAL_72]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:             %[[VAL_78:.*]] = ttg.memdesc_trans %[[VAL_76]] {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64> -> !ttg.memdesc<64x256xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable>
// CHECK:             %[[VAL_79:.*]] = ttng.warp_group_dot %[[VAL_77]], %[[VAL_78]], %[[VAL_63]] {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64> * !ttg.memdesc<64x256xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable> -> tensor<128x256xf32, #[[$ATTR_1]]>
// CHECK:             %[[VAL_80:.*]]:3 = ttng.warp_group_dot_wait %[[VAL_79]], %[[VAL_77]], %[[VAL_78]] {pendings = 1 : i32} : tensor<128x256xf32, #[[$ATTR_1]]>, !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable>
// CHECK:             %[[VAL_81:.*]] = arith.addi %[[VAL_64]], %[[VAL_13]] : i32
// CHECK:             %[[VAL_82:.*]] = arith.addi %[[VAL_65]], %[[VAL_15]] : i32
// CHECK:             %[[VAL_83:.*]] = arith.cmpi slt, %[[VAL_82]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_84:.*]] = arith.select %[[VAL_83]], %[[VAL_82]], %[[VAL_12]] : i32
// CHECK:             %[[VAL_85:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_84]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:             ttng.barrier_expect %[[VAL_85]], 49152, %[[VAL_69]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:             %[[VAL_86:.*]] = ttg.memdesc_subview %[[VAL_43]]{{\[}}%[[VAL_84]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:             %[[VAL_87:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_35]] : !tt.tensordesc<tensor<128x64xf16>> to !tt.ptr<i8>
// CHECK:             ttng.async_tma_copy_global_to_local %[[VAL_87]]{{\[}}%[[VAL_39]], %[[VAL_81]]] %[[VAL_86]], %[[VAL_85]], %[[VAL_69]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x128x64>
// CHECK:             %[[VAL_88:.*]] = ttg.memdesc_subview %[[VAL_44]]{{\[}}%[[VAL_84]], %[[VAL_12]], %[[VAL_12]]] : !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:             %[[VAL_89:.*]] = ttng.tensor_desc_to_tma_ptr %[[VAL_36]] : !tt.tensordesc<tensor<256x64xf16>> to !tt.ptr<i8>
// CHECK:             ttng.async_tma_copy_global_to_local %[[VAL_89]]{{\[}}%[[VAL_40]], %[[VAL_81]]] %[[VAL_88]], %[[VAL_85]], %[[VAL_69]] : <i8>, <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3> -> <256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable, 3x256x64>
// CHECK:             scf.yield %[[VAL_80]]#0, %[[VAL_81]], %[[VAL_84]], %[[VAL_72]], %[[VAL_74]] : tensor<128x256xf32, #[[$ATTR_1]]>, i32, i32, i32, i32
// CHECK:           }
// CHECK:           %[[VAL_90:.*]] = ttng.warp_group_dot_wait %[[VAL_91:.*]]#0 {pendings = 0 : i32} : tensor<128x256xf32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_92:.*]] = ttg.async_wait  {num = 0 : i32}
// CHECK:           %[[VAL_93:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_12]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.inval_barrier %[[VAL_93]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_94:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_15]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.inval_barrier %[[VAL_94]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           %[[VAL_95:.*]] = ttg.memdesc_subview %[[VAL_45]]{{\[}}%[[VAL_6]]] : !ttg.memdesc<3xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> !ttg.memdesc<1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttng.inval_barrier %[[VAL_95]] : <1xi64, #[[$ATTR_3]], #[[$ATTR_5]], mutable, 3>
// CHECK:           ttg.local_dealloc %[[VAL_43]] : !ttg.memdesc<3x128x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable>
// CHECK:           ttg.local_dealloc %[[VAL_44]] : !ttg.memdesc<3x256x64xf16, #[[$ATTR_2]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_96:.*]] = arith.truncf %[[VAL_90]] : tensor<128x256xf32, #[[$ATTR_1]]> to tensor<128x256xf16, #[[$ATTR_1]]>
// CHECK:           %[[VAL_97:.*]] = ttg.convert_layout %[[VAL_96]] : tensor<128x256xf16, #[[$ATTR_1]]> -> tensor<128x256xf16, #[[$ATTR_0]]>
// CHECK:           tt.experimental_descriptor_store %[[VAL_38]]{{\[}}%[[VAL_39]], %[[VAL_40]]], %[[VAL_97]] : !tt.tensordesc<tensor<128x256xf16>>, tensor<128x256xf16, #[[$ATTR_0]]>
// CHECK:           tt.return
// CHECK:         }
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_with_descriptors(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.extsi %arg5 : i32 to i64
    %15 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%14, %c1_i64] : <f16>, <tensor<128x64xf16>>
    %16 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%14, %c1_i64] : <f16>, <tensor<256x64xf16>>
    %17 = arith.extsi %arg4 : i32 to i64
    %18 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%17, %c1_i64] : <f16>, <tensor<128x256xf16>>
    %19 = arith.muli %11, %c128_i32 : i32
    %20 = arith.muli %13, %c256_i32 : i32
    %21 = arith.addi %arg5, %c63_i32 : i32
    %22 = arith.divsi %21, %c64_i32 : i32
    %23:2 = scf.for %arg6 = %c0_i32 to %22 step %c1_i32 iter_args(%arg7 = %cst, %arg8 = %c0_i32) -> (tensor<128x256xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>>, i32)  : i32 {
      %26 = tt.experimental_descriptor_load %15[%19, %arg8] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>
      %27 = ttg.local_alloc %26 : (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>, #ttg.shared_memory>
      %28 = tt.experimental_descriptor_load %16[%20, %arg8] : !tt.tensordesc<tensor<256x64xf16>> -> tensor<256x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>
      %29 = ttg.local_alloc %28 : (tensor<256x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>) -> !ttg.memdesc<256x64xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>, #ttg.shared_memory>
      %30 = ttg.memdesc_trans %29 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>, #ttg.shared_memory> -> !ttg.memdesc<64x256xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>, #ttg.shared_memory>
      %31 = ttng.warp_group_dot %27, %30, %arg7 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>, #ttg.shared_memory> * !ttg.memdesc<64x256xf16, #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>, #ttg.shared_memory> -> tensor<128x256xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>>
      %32 = arith.addi %arg8, %c64_i32 : i32
      scf.yield %31, %32 : tensor<128x256xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>>, i32
    }
    %24 = arith.truncf %23#0 : tensor<128x256xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>> to tensor<128x256xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>>
    %25 = ttg.convert_layout %24 : tensor<128x256xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>> -> tensor<128x256xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>>
    tt.experimental_descriptor_store %18[%19, %20], %25 : !tt.tensordesc<tensor<128x256xf16>>, tensor<128x256xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>>
    tt.return
  }
}