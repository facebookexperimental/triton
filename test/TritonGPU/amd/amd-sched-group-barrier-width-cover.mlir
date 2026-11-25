// RUN: triton-opt %s -split-input-file --tritonamdgpu-sched-group-barrier-scheduler | FileCheck %s

// The VMEM cover scales with the access width that final TTGIR will lower:
// dwordx4 -> 4 MFMA, dwordx2 -> 2 MFMA, dword -> 1 MFMA.

// CHECK-LABEL: @dwordx4
// CHECK: amdg.buffer_load
// CHECK-SAME: machine_count = 1 : i32
// CHECK-SAME: machine_mask = 32 : i32
// CHECK-SAME: mfma_cover = 4 : i32
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dwordx4(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32}) {
    %result = amdg.buffer_load %ptr[%offsets] : !tt.ptr<f32> -> tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

// One gfx1250 block-level FP4 dot lowers to four native WMMA instructions per
// wave: N256 / (N16 * 4 N-distributed warps) = 4. Its B operand load is 128 B
// per lane, read with ds_read_b128 (the MFMA B-operand ds_read_b64 rule does
// not apply to WMMA): 8 DS reads.

// CHECK-LABEL: @wmma_fp4
// CHECK: ttg.local_load
// CHECK-SAME: machine_count = 8 : i32
// CHECK-SAME: machine_mask = 256 : i32
// CHECK: tt.dot_scaled
// CHECK-SAME: machine_count = 4 : i32
// CHECK-SAME: machine_mask = 8 : i32
#scale_a = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [0, 2]]}, isTranspose = true, instrShape = [32, 16, 128]}>
#operand_mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [0, 2]]}, isTranspose = true, instrShape = [32, 16, 64]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #operand_mma, kWidth = 16}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #operand_mma, kWidth = 16}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_fp4(
      %a: tensor<32x64xi8, #lhs>,
      %a_scale: tensor<32x4xi8, #scale_a>,
      %b_smem: !ttg.memdesc<64x256xi8, #shared, #smem>,
      %b_scale: tensor<256x4xi8, #scale_b>,
      %acc: tensor<32x256xf32, #mma>) {
    %b = ttg.local_load %b_smem : !ttg.memdesc<64x256xi8, #shared, #smem> -> tensor<64x256xi8, #rhs>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc
        lhs = e2m1 rhs = e2m1 {fastMath = false}
        : tensor<32x64xi8, #lhs>, tensor<32x4xi8, #scale_a>
        * tensor<64x256xi8, #rhs>, tensor<256x4xi8, #scale_b>
        -> tensor<32x256xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: @dwordx2
// CHECK: amdg.buffer_load
// CHECK-SAME: machine_count = 1 : i32
// CHECK-SAME: machine_mask = 32 : i32
// CHECK-SAME: mfma_cover = 2 : i32
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dwordx2(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<512xi32, #blocked> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}) {
    %result = amdg.buffer_load %ptr[%offsets] : !tt.ptr<f32> -> tensor<512xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: @dword
// CHECK: amdg.buffer_load
// CHECK-SAME: machine_count = 1 : i32
// CHECK-SAME: machine_mask = 32 : i32
// CHECK-SAME: mfma_cover = 1 : i32
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dword(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<256xi32, #blocked> {tt.contiguity = 1 : i32, tt.divisibility = 16 : i32}) {
    %result = amdg.buffer_load %ptr[%offsets] : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}
