// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_24
  // CHECK: [[N24:%.*]] = llvm.mlir.constant(24 : i32)
  // CHECK: [[OWNER24:%.*]] = llvm.icmp "ult" {{.*}}, [[N24]] : i32
  // The canonical mask feeds AMD's arbitrary-mask intra-wave compaction.
  // CHECK: [[CANON24:%.*]] = llvm.and {{%.*}}, [[OWNER24]] : i1
  // CHECK: llvm.zext [[CANON24]] : i1 to i32
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.ds.permute"
  // CHECK: llvm.atomicrmw add
  tt.func @npot_atomic_rmw_24(%ptr: tensor<24x!tt.ptr<i32>, #blocked>, %val: tensor<24xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<24x!tt.ptr<i32>, #blocked>, tensor<24xi32, #blocked>) -> tensor<24xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_64x48
  // CHECK: [[N2D:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER2D:%.*]] = llvm.icmp "ult" {{.*}}, [[N2D]] : i32
  // CHECK: [[CANON2D:%.*]] = llvm.and {{%.*}}, [[OWNER2D]] : i1
  // CHECK: llvm.zext [[CANON2D]] : i1 to i32
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.ds.permute"
  // CHECK: llvm.atomicrmw add
  tt.func @npot_atomic_rmw_64x48(%ptr: tensor<64x48x!tt.ptr<i32>, #blocked>, %val: tensor<64x48xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<64x48x!tt.ptr<i32>, #blocked>, tensor<64x48xi32, #blocked>) -> tensor<64x48xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: llvm.func @npot_atomic_cas_80
  // CHECK: [[N80:%.*]] = llvm.mlir.constant(80 : i32)
  // CHECK: [[OWNER80:%.*]] = llvm.icmp "ult" {{.*}}, [[N80]] : i32
  // CHECK: [[CANON80:%.*]] = llvm.and {{%.*}}, [[OWNER80]] : i1
  // CHECK: llvm.cond_br [[CANON80]]
  // CHECK: llvm.cmpxchg
  tt.func @npot_atomic_cas_80(%ptr: tensor<80x!tt.ptr<i32>, #blocked>, %cmp: tensor<80xi32, #blocked>, %val: tensor<80xi32, #blocked>) {
    %0 = tt.atomic_cas relaxed, gpu, %ptr, %cmp, %val : (tensor<80x!tt.ptr<i32>, #blocked>, tensor<80xi32, #blocked>, tensor<80xi32, #blocked>) -> tensor<80xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: llvm.func @npot_buffer_atomic_rmw_masked_48
  // CHECK: [[N48:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER48:%.*]] = llvm.icmp "ult" {{.*}}, [[N48]] : i32
  // CHECK: [[CANON48:%.*]] = llvm.and {{%.*}}, [[OWNER48]] : i1
  // CHECK: [[MASKED48:%.*]] = llvm.and [[CANON48]], {{%.*}} : i1
  // CHECK: llvm.select [[MASKED48]]
  // CHECK: buffer.atomic
  tt.func @npot_buffer_atomic_rmw_masked_48(%ptr: !tt.ptr<i32>, %offsets: tensor<48xi32, #blocked>, %val: tensor<48xi32, #blocked>, %mask: tensor<48xi1, #blocked>) {
    %0 = amdg.buffer_atomic_rmw add, relaxed, gpu, %val, %ptr[%offsets], %mask : tensor<48xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: llvm.func @npot_buffer_atomic_cas_96
  // CHECK: [[N96:%.*]] = llvm.mlir.constant(96 : i32)
  // CHECK: [[OWNER96:%.*]] = llvm.icmp "ult" {{.*}}, [[N96]] : i32
  // CHECK: [[CANON96:%.*]] = llvm.and {{%.*}}, [[OWNER96]] : i1
  // CHECK: llvm.select [[CANON96]]
  // CHECK: buffer.atomic.cmpswap
  tt.func @npot_buffer_atomic_cas_96(%ptr: !tt.ptr<i32>, %offsets: tensor<96xi32, #blocked>, %cmp: tensor<96xi32, #blocked>, %val: tensor<96xi32, #blocked>) {
    %0 = amdg.buffer_atomic_cas relaxed, gpu, %cmp, %val, %ptr[%offsets] : tensor<96xi32, #blocked>
    tt.return
  }
}
