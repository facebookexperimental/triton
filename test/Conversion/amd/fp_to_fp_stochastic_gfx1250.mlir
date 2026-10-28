// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s
//
// Stochastic rounding (rounding = rs) on CDNA5 (gfx1250) lowers f32 to OCP
// fp8 (V_CVT_SR_FP8/BF8_F32 via rocdl.cvt.sr.*) and to f16/bf16 with the
// packed converts (V_CVT_SR_PK_F16/BF16_F32 via llvm.amdgcn.cvt.sr.pk.*
// intrinsics).

// CHECK-LABEL: f32_to_f8_stochastic
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f8_stochastic(%arg0: tensor<512xf32, #blocked>,
                                %rbits: tensor<512xi32, #blocked>) {
    // Four elements per thread pack into one i32 through a byte-select
    // chain: each convert reads the previous convert's accumulator.
    // CHECK: %[[B0:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0] : i32
    // CHECK: %[[B1:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B0]][1] : i32
    // CHECK: %[[B2:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B1]][2] : i32
    // CHECK: %[[B3:.*]] = rocdl.cvt.sr.bf8.f32 %{{.*}}, %{{.*}} -> %[[B2]][3] : i32
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<512xi32, #blocked>, rounding = rs : tensor<512xf32, #blocked> -> tensor<512xf8E5M2, #blocked>
    // CHECK: %[[F0:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[0] : i32
    // CHECK: %[[F1:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F0]][1] : i32
    // CHECK: %[[F2:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F1]][2] : i32
    // CHECK: %[[F3:.*]] = rocdl.cvt.sr.fp8.f32 %{{.*}}, %{{.*}} -> %[[F2]][3] : i32
    // CHECK-NOT: rocdl.cvt.sr.
    // CHECK: llvm.return
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<512xi32, #blocked>, rounding = rs : tensor<512xf32, #blocked> -> tensor<512xf8E4M3FN, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: f32_to_f16_bf16_stochastic
#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f16_bf16_stochastic(%arg0: tensor<512xf32, #blocked2>,
                                      %rbits: tensor<512xi32, #blocked2>) {
    // Four elements per thread: two packed converts. Each packs two
    // per-element seeds into one i32 (low lane | high lane << 16); f16
    // lanes are masked to the 13 discarded mantissa bits (8191). The
    // hardware adds the low lane to the first f32 and the high lane to
    // the second (CDNA5 ISA, V_CVT_SR_PK_F16_F32), so the packing must
    // feed the intrinsic as its third operand.
    // CHECK: %[[MASK16:.*]] = llvm.mlir.constant(8191 : i32) : i32
    // CHECK: %[[LO0:.*]] = llvm.and %{{.*}}, %[[MASK16]]
    // CHECK: %[[HI0:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[SH0:.*]] = llvm.shl %[[HI0]], %{{.*}}
    // CHECK: %[[SEED0:.*]] = llvm.or %[[LO0]], %[[SH0]]
    // CHECK: %[[P0:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.sr.pk.f16.f32"(%{{.*}}, %{{.*}}, %[[SEED0]]) : (f32, f32, i32) -> vector<2xf16>
    // CHECK: %[[LO1:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[HI1:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[SH1:.*]] = llvm.shl %[[HI1]], %{{.*}}
    // CHECK: %[[SEED1:.*]] = llvm.or %[[LO1]], %[[SH1]]
    // CHECK: %[[P1:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.sr.pk.f16.f32"(%{{.*}}, %{{.*}}, %[[SEED1]]) : (f32, f32, i32) -> vector<2xf16>
    %0 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<512xi32, #blocked2>, rounding = rs : tensor<512xf32, #blocked2> -> tensor<512xf16, #blocked2>
    // bf16 lanes keep the full 16 discarded bits (65535).
    // CHECK: %[[MASKBF:.*]] = llvm.mlir.constant(65535 : i32) : i32
    // CHECK: %[[BLO0:.*]] = llvm.and %{{.*}}, %[[MASKBF]]
    // CHECK: %[[BHI0:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[BSH0:.*]] = llvm.shl %[[BHI0]], %{{.*}}
    // CHECK: %[[BSEED0:.*]] = llvm.or %[[BLO0]], %[[BSH0]]
    // CHECK: %[[Q0:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.sr.pk.bf16.f32"(%{{.*}}, %{{.*}}, %[[BSEED0]]) : (f32, f32, i32) -> vector<2xbf16>
    // CHECK: %[[BLO1:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[BHI1:.*]] = llvm.and %{{.*}}, %{{.*}}
    // CHECK: %[[BSH1:.*]] = llvm.shl %[[BHI1]], %{{.*}}
    // CHECK: %[[BSEED1:.*]] = llvm.or %[[BLO1]], %[[BSH1]]
    // CHECK: %[[Q1:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.sr.pk.bf16.f32"(%{{.*}}, %{{.*}}, %[[BSEED1]]) : (f32, f32, i32) -> vector<2xbf16>
    // CHECK-NOT: llvm.amdgcn.cvt.sr.pk.
    // CHECK: llvm.return
    %1 = tt.fp_to_fp %arg0, rbits = %rbits : tensor<512xi32, #blocked2>, rounding = rs : tensor<512xf32, #blocked2> -> tensor<512xbf16, #blocked2>
    tt.return
  }
}
