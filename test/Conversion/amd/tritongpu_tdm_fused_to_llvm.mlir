// RUN: triton-opt %s --split-input-file | FileCheck %s --check-prefix=TTG
// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM
// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm --canonicalize --cse | FileCheck %s --check-prefix=FOLD

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_explicit_multicast
  // LLVM-LABEL: tdm_fused_explicit_multicast
  // FOLD-LABEL: tdm_fused_explicit_multicast
  tt.func public @tdm_fused_explicit_multicast(
      %a: !tt.tensordesc<64x64xf16, #shared>,
      %b: !tt.tensordesc<64x64xf16, #shared>,
      %da: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %db: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %mask_a: i32, %mask_b: i32) {
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: multicast %{{.*}}, %{{.*}}
    // LLVM: llvm.or %{{.*}}, %arg4 : i32
    // LLVM: llvm.or %{{.*}}, %arg5 : i32
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // FOLD: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db multicast %mask_a, %mask_b {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_distributed_explicit_multicast
  // LLVM-LABEL: tdm_fused_distributed_explicit_multicast
  // FOLD-LABEL: tdm_fused_distributed_explicit_multicast
  tt.func public @tdm_fused_distributed_explicit_multicast(
      %a: !tt.tensordesc<64x64xf16, #shared>,
      %b: !tt.tensordesc<64x64xf16, #shared>,
      %da: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %db: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %mask_a: i32, %mask_b: i32) {
    // Explicit masks replace the inferred {0,2}/{1,3} recipient groups while
    // the layout still determines the per-CTA column partition.
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: multicast %{{.*}}, %{{.*}}
    // LLVM: rocdl.cluster.workgroup.id.x
    // LLVM: llvm.or %{{.*}}, %arg4 : i32
    // LLVM: llvm.or %{{.*}}, %arg5 : i32
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // FOLD: rocdl.cluster.workgroup.id.x
    // FOLD: llvm.getelementptr
    // FOLD: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db multicast %mask_a, %mask_b {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_manual_hints_stay_separate
  // LLVM-LABEL: tdm_manual_hints_stay_separate
  tt.func public @tdm_manual_hints_stay_separate(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc0_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : !tt.ptr<f16>, !tt.tensordesc<64x64xf16, #shared>
    %desc1_base = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : !tt.ptr<f16>, !tt.tensordesc<64x64xf16, #shared>
    %desc0 = amdg.update_tensor_descriptor %desc0_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %desc1 = amdg.update_tensor_descriptor %desc1_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // Explicit hints on regular copies do not request implicit fusion.
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-SAME: warp_used_hint = 3 : i32
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-SAME: warp_used_hint = 12 : i32
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_copy_global_to_local %desc0 into %dst0 {warp_used_hint = 3 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc1 into %dst1 {warp_used_hint = 12 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_fold_position
  // LLVM-LABEL: tdm_fused_fold_position
  // FOLD-LABEL: llvm.func @tdm_fused_fold_position
  tt.func public @tdm_fused_fold_position(
      %desc: !tt.tensordesc<32x32xf16, #shared>,
      %offset: i32,
      %da: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>,
      %db: !ttg.memdesc<32x32xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    // Share the positioned descriptor between two destinations. Its address
    // must not be updated/repacked before the copy stamps its LDS address.
    // The dynamic offset retains the signed descriptor-update semantics.
    // FOLD-NOT: llvm.bitcast
    // FOLD: llvm.sext %arg1 : i32 to i64
    // FOLD-NOT: llvm.bitcast
    // FOLD: llvm.ptrtoint
    // FOLD: llvm.call_intrinsic "llvm.amdgcn.tensor.load.to.lds"
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%c0, %offset] : !tt.tensordesc<32x32xf16, #shared>
    %0 = amdg.async_tdm_fused_copy_global_to_local %positioned, %positioned into %da, %db {warp_used_hints = array<i32: 1, 2>} : !tt.tensordesc<32x32xf16, #shared>, !tt.tensordesc<32x32xf16, #shared> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_heterogeneous_members
  // LLVM-LABEL: tdm_fused_heterogeneous_members
  tt.func public @tdm_fused_heterogeneous_members(
      %a: !tt.tensordesc<64x64xf16, #shared>,
      %b: !tt.tensordesc<32x32xf32, #shared>,
      %c: !tt.tensordesc<128x16xf16, #shared>,
      %da: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %db: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
      %dc: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>) {
    // Hints may leave warps unused, and members may differ in shape and type.
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: warp_used_hints = array<i32: 1, 2, 4>
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b, %c into %da, %db, %dc {warp_used_hints = array<i32: 1, 2, 4>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<32x32xf32, #shared>, !tt.tensordesc<128x16xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_explicit_fused
  // LLVM-LABEL: tdm_explicit_fused
  tt.func public @tdm_explicit_fused(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc0_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : !tt.ptr<f16>, !tt.tensordesc<64x64xf16, #shared>
    %desc1_base = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : !tt.ptr<f16>, !tt.tensordesc<64x64xf16, #shared>
    %desc0 = amdg.update_tensor_descriptor %desc0_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %desc1 = amdg.update_tensor_descriptor %desc1_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: warp_used_hints = array<i32: 3, 12>
    // TTG-NOT: amdg.async_tdm_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
