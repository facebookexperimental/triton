// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK: ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_24
  // CHECK: [[N24:%.*]] = llvm.mlir.constant(24 : i32)
  // CHECK: [[OWNER24:%.*]] = llvm.icmp "ult" {{.*}}, [[N24]] : i32
  // CHECK: [[CANON24:%.*]] = llvm.and {{%.*}}, [[OWNER24]] : i1
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,b" {{.*}}, [[CANON24]] :
  tt.func @npot_atomic_rmw_24(%ptr: tensor<24x!tt.ptr<i32>, #blocked>, %val: tensor<24xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<24x!tt.ptr<i32>, #blocked>, tensor<24xi32, #blocked>) -> tensor<24xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_masked_48
  // CHECK: [[N48:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER48:%.*]] = llvm.icmp "ult" {{.*}}, [[N48]] : i32
  // CHECK: [[CANON48:%.*]] = llvm.and {{%.*}}, [[OWNER48]] : i1
  // CHECK: [[MASKED48:%.*]] = llvm.and [[CANON48]], {{%.*}} : i1
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,b" {{.*}}, [[MASKED48]] :
  tt.func @npot_atomic_rmw_masked_48(%ptr: tensor<48x!tt.ptr<i32>, #blocked>, %val: tensor<48xi32, #blocked>, %mask: tensor<48xi1, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val, %mask : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>, tensor<48xi1, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_atomic_cas_80
  // CHECK: [[N80:%.*]] = llvm.mlir.constant(80 : i32)
  // CHECK: [[OWNER80:%.*]] = llvm.icmp "ult" {{.*}}, [[N80]] : i32
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,r,b" {{.*}}, [[OWNER80]] :
  tt.func @npot_atomic_cas_80(%ptr: tensor<80x!tt.ptr<i32>, #blocked>, %cmp: tensor<80xi32, #blocked>, %val: tensor<80xi32, #blocked>) {
    %0 = tt.atomic_cas relaxed, gpu, %ptr, %cmp, %val : (tensor<80x!tt.ptr<i32>, #blocked>, tensor<80xi32, #blocked>, tensor<80xi32, #blocked>) -> tensor<80xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_96
  // CHECK: [[N96:%.*]] = llvm.mlir.constant(96 : i32)
  // CHECK: [[OWNER96:%.*]] = llvm.icmp "ult" {{.*}}, [[N96]] : i32
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,b" {{.*}}, [[OWNER96]] :
  tt.func @npot_atomic_rmw_96(%ptr: tensor<96x!tt.ptr<i32>, #blocked>, %val: tensor<96xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<96x!tt.ptr<i32>, #blocked>, tensor<96xi32, #blocked>) -> tensor<96xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK: ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_64x48
  // CHECK: [[N2D:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER2D:%.*]] = llvm.icmp "ult" {{.*}}, [[N2D]] : i32
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,b" {{.*}}, [[OWNER2D]] :
  tt.func @npot_atomic_rmw_64x48(%ptr: tensor<64x48x!tt.ptr<i32>, #blocked>, %val: tensor<64x48xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<64x48x!tt.ptr<i32>, #blocked>, tensor<64x48xi32, #blocked>) -> tensor<64x48xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:90"} {
  // CHECK-LABEL: llvm.func @npot_atomic_rmw_2cta
  // CHECK: [[CTA_BOUND:%.*]] = llvm.mlir.constant(24 : i32)
  // CHECK: [[CTA_OWNER:%.*]] = llvm.icmp "ult" {{.*}}, [[CTA_BOUND]] : i32
  // CHECK: [[CTA_CANON:%.*]] = llvm.and {{%.*}}, [[CTA_OWNER]] : i1
  // CHECK: llvm.inline_asm {{.*}} "=r,l,r,b" {{.*}}, [[CTA_CANON]] :
  tt.func @npot_atomic_rmw_2cta(%ptr: tensor<48x!tt.ptr<i32>, #blocked>, %val: tensor<48xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}
