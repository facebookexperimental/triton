// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --allocate-shared-memory-nv --convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_gsan_atomic_rmw_48
  // CHECK: [[N48:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER48:%.*]] = llvm.icmp "ult" {{.*}}, [[N48]] : i32
  // CHECK: [[CANON48:%.*]] = llvm.and {{%.*}}, [[OWNER48]] : i1
  // CHECK: [[CANON48_I32:%.*]] = llvm.zext [[CANON48]] : i1 to i32
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar({{.*}}, [[CANON48_I32]], {{.*}})
  // CHECK: llvm.inline_asm {{.*}} [[CANON48]]
  // CHECK: [[CANON48_END_I32:%.*]] = llvm.zext [[CANON48]] : i1 to i32
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar({{.*}}, [[CANON48_END_I32]], {{.*}})
  tt.func @npot_gsan_atomic_rmw_48(%ptr: tensor<48x!tt.ptr<i32>, #blocked>, %val: tensor<48xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: llvm.func @npot_gsan_atomic_cas_64x48
  // CHECK: [[N2D:%.*]] = llvm.mlir.constant(48 : i32)
  // CHECK: [[OWNER2D:%.*]] = llvm.icmp "ult" {{.*}}, [[N2D]] : i32
  // CHECK: [[OWNER2D_I32:%.*]] = llvm.zext [[OWNER2D]] : i1 to i32
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar({{.*}}, [[OWNER2D_I32]], {{.*}})
  // CHECK: llvm.inline_asm {{.*}} [[OWNER2D]]
  // CHECK: [[OWNER2D_END_I32:%.*]] = llvm.zext [[OWNER2D]] : i1 to i32
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar({{.*}}, [[OWNER2D_END_I32]], {{.*}})
  tt.func @npot_gsan_atomic_cas_64x48(%ptr: tensor<64x48x!tt.ptr<i32>, #blocked>, %cmp: tensor<64x48xi32, #blocked>, %val: tensor<64x48xi32, #blocked>) {
    %0 = tt.atomic_cas relaxed, gpu, %ptr, %cmp, %val : (tensor<64x48x!tt.ptr<i32>, #blocked>, tensor<64x48xi32, #blocked>, tensor<64x48xi32, #blocked>) -> tensor<64x48xi32, #blocked>
    tt.return
  }
}
