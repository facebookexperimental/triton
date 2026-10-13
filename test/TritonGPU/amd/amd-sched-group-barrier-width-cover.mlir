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
    %result = amdg.buffer_load %ptr[%offsets] : tensor<1024xf32, #blocked>
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
    %result = amdg.buffer_load %ptr[%offsets] : tensor<512xf32, #blocked>
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
    %result = amdg.buffer_load %ptr[%offsets] : tensor<256xf32, #blocked>
    tt.return
  }
}
