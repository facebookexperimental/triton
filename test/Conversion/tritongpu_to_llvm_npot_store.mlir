// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s

// A modular (NPOT) tensor rounds its register span up to a power of two, so the
// phantom registers get raw out-of-range pointer offsets. A maskless store must
// predicate every register to a single canonical owner, otherwise the phantom
// registers overflow into the next logical element.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK: ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @npot_store_24
  // CHECK: [[N24:%.*]] = llvm.mlir.constant(24 : i32)
  // CHECK: [[OWNER24:%.*]] = llvm.icmp "ult" {{.*}}, [[N24]] : i32
  // CHECK: [[CANON24:%.*]] = llvm.and {{%.*}}, [[OWNER24]] : i1
  // CHECK: llvm.inline_asm {{.*}} "r,l,b" {{.*}}, [[CANON24]] :
  tt.func @npot_store_24(%ptr: tensor<24x!tt.ptr<i32>, #blocked>, %val: tensor<24xi32, #blocked>) {
    tt.store %ptr, %val : tensor<24x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_store_masked_48
  // CHECK: [[N48:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER48:%.*]] = llvm.icmp "ult" {{.*}}, [[N48]] : i32
  // CHECK: [[CANON48:%.*]] = llvm.and {{%.*}}, [[OWNER48]] : i1
  // CHECK: [[MASKED48:%.*]] = llvm.and [[CANON48]], {{%.*}} : i1
  // CHECK: llvm.inline_asm {{.*}} "r,l,b" {{.*}}, [[MASKED48]] :
  tt.func @npot_store_masked_48(%ptr: tensor<48x!tt.ptr<i32>, #blocked>, %val: tensor<48xi32, #blocked>, %mask: tensor<48xi1, #blocked>) {
    tt.store %ptr, %val, %mask : tensor<48x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// Rectangular 2D layout: the NPOT column dim (48) rounds up to 64, so the
// phantom columns must be predicated by their logical column bound.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK: ttg.shared = 0 : i32
  // CHECK-LABEL: llvm.func @npot_store_64x48
  // CHECK: [[N2D:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER2D:%.*]] = llvm.icmp "ult" {{.*}}, [[N2D]] : i32
  // CHECK: llvm.inline_asm {{.*}} "r,l,b" {{.*}}, {{%.*}} :
  tt.func @npot_store_64x48(%ptr: tensor<64x48x!tt.ptr<i32>, #blocked>, %val: tensor<64x48xi32, #blocked>) {
    tt.store %ptr, %val : tensor<64x48x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
