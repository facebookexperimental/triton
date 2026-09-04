// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s
// RUN: triton-opt %s -split-input-file --cse | FileCheck %s --check-prefix=CSE

// CHECK-LABEL:mfma_16x16x32_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_f16(%arg0: tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_16x16x32_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf16(%arg0: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_f16(%arg0: tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}


// -----

// CHECK-LABEL:mfma_32x32x16_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf16(%arg0: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                         %arg1: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
 }
}

// -----

// When kWidth is set to 4, still generate double rated mfma instructions.

// CHECK-LABEL:mfma_16x16x32_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_16x16x32_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_f16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} : (vector<8xf16>, vector<8xf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mfma_32x32x16_bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK: rocdl.mfma.f32.32x32x16.bf16 {{.*}} : (vector<8xbf16>, vector<8xbf16>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL:mxfp4_2step
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mxfp4_2step(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK-COUNT-32: rocdl.mfma.scale.f32.16x16x128.f8f6f4
    // CHECK: rocdl.sched.barrier none
    // CHECK: rocdl.s.barrier
    // CHECK: rocdl.sched.barrier none
    // CHECK-COUNT-32: rocdl.mfma.scale.f32.16x16x128.f8f6f4
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false, pingpong_2step} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
 }
}

// -----

// CHECK-LABEL: llvm.func @scheduled_mfma_source_control
// CHECK: llvm.inline_asm
// CHECK-SAME: "=a,0"
// CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf32>)
// CHECK: rocdl.mfma.f32.16x16x32.bf16 {{.*}}, {{.*}}, %[[ZERO]], 0, 0, none
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5"
// CHECK-SAME: "=v,=a,0,1,~{memory}"
// CHECK-NOT: amdg.

#scheduled_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_mma, kWidth = 8}>
#scheduled_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_source_control(
      %a: tensor<16x32xbf16, #scheduled_lhs>,
      %b: tensor<32x16xbf16, #scheduled_rhs>) {
    %resident_b = amdg.register_resident %b class "agpr" groups 4
        : tensor<32x16xbf16, #scheduled_rhs>
    %acc = arith.constant dense<7.000000e+00> :
        tensor<16x16xf32, #scheduled_mma>
    %result = amdg.scheduled_mfma %a, %resident_b, %acc
        resident "rhs" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #scheduled_lhs>,
          tensor<32x16xbf16, #scheduled_rhs>,
          tensor<16x16xf32, #scheduled_mma>
          -> tensor<16x16xf32, #scheduled_mma>
    %committed, %preserved = amdg.mfma_commit %result, %resident_b
        : tensor<16x16xf32, #scheduled_mma>,
          tensor<32x16xbf16, #scheduled_rhs>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @scheduled_mfma_kwidth4
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5"
// CHECK-NOT: amdg.

#scheduled_k4_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_k4_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_k4_mma, kWidth = 4}>
#scheduled_k4_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_k4_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_kwidth4(
      %a: tensor<16x32xbf16, #scheduled_k4_lhs>,
      %b: tensor<32x16xbf16, #scheduled_k4_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #scheduled_k4_mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #scheduled_k4_lhs>,
          tensor<32x16xbf16, #scheduled_k4_rhs>,
          tensor<16x16xf32, #scheduled_k4_mma>
          -> tensor<16x16xf32, #scheduled_k4_mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #scheduled_k4_mma>,
          tensor<32x16xbf16, #scheduled_k4_rhs>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @scheduled_mfma_32x32_kwidth4
// CHECK: rocdl.mfma.f32.32x32x16.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5"
// CHECK-NOT: amdg.

#scheduled_32_k4_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#scheduled_32_k4_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_32_k4_mma, kWidth = 4}>
#scheduled_32_k4_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_32_k4_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_32x32_kwidth4(
      %a: tensor<32x16xbf16, #scheduled_32_k4_lhs>,
      %b: tensor<16x32xbf16, #scheduled_32_k4_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<32x32xf32, #scheduled_32_k4_mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<32x16xbf16, #scheduled_32_k4_lhs>,
          tensor<16x32xbf16, #scheduled_32_k4_rhs>,
          tensor<32x32xf32, #scheduled_32_k4_mma>
          -> tensor<32x32xf32, #scheduled_32_k4_mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<32x32xf32, #scheduled_32_k4_mma>,
          tensor<16x32xbf16, #scheduled_32_k4_rhs>
    tt.return
  }
}

// -----

// K-major lowering initializes every independent output fragment before
// returning to any accumulator chain for the second native K slice.
// CHECK-LABEL: llvm.func @scheduled_mfma_round_robin_k
// CHECK-COUNT-4: llvm.inline_asm has_side_effects{{.*}}"s_nop 3\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0"
// CHECK-COUNT-4: llvm.inline_asm has_side_effects{{.*}}"s_nop 3\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0"
// CHECK-COUNT-4: llvm.inline_asm has_side_effects{{.*}}"s_nop 11"
// CHECK-NOT: amdg.

#round_robin_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#round_robin_lhs = #ttg.dot_op<{opIdx = 0, parent = #round_robin_mma, kWidth = 8}>
#round_robin_rhs = #ttg.dot_op<{opIdx = 1, parent = #round_robin_mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_round_robin_k(
      %a: tensor<32x64xbf16, #round_robin_lhs>,
      %b: tensor<64x32xbf16, #round_robin_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<32x32xf32, #round_robin_mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x64xbf16, #round_robin_lhs>,
          tensor<64x32xbf16, #round_robin_rhs>,
          tensor<32x32xf32, #round_robin_mma>
          -> tensor<32x32xf32, #round_robin_mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @register_class_anchor_vgpr
// CSE-LABEL: tt.func public @register_class_anchor_vgpr
// CSE-COUNT-2: amdg.register_class_anchor
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=v,0"
// CHECK-NOT: amdg.register_class_anchor

#anchor_fp32 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @register_class_anchor_vgpr(
      %arg: tensor<256xf32, #anchor_fp32>) -> tensor<256xf32, #anchor_fp32> {
    %result = amdg.register_class_anchor %arg class "vgpr"
        : tensor<256xf32, #anchor_fp32>
    %independent = amdg.register_class_anchor %arg class "vgpr"
        : tensor<256xf32, #anchor_fp32>
    %sum = arith.addf %result, %independent : tensor<256xf32, #anchor_fp32>
    tt.return %sum : tensor<256xf32, #anchor_fp32>
  }
}

// -----

// CHECK-LABEL: llvm.func @register_class_anchor_packs_fp16_registers
// CHECK: llvm.inline_asm
// CHECK-SAME: "=a,0"
// CHECK: llvm.inline_asm
// CHECK-SAME: "=a,0"
// CHECK-NOT: amdg.register_class_anchor

#anchor_fp16 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @register_class_anchor_packs_fp16_registers(
      %arg: tensor<256xf16, #anchor_fp16>) -> tensor<256xf16, #anchor_fp16> {
    %result = amdg.register_class_anchor %arg class "agpr"
        : tensor<256xf16, #anchor_fp16>
    tt.return %result : tensor<256xf16, #anchor_fp16>
  }
}
