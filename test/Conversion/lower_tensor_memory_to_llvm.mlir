// RUN: triton-opt %s -split-input-file --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.tensor_memory_size = 128 : i32, "ttng.two-ctas" = true} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: @automatic_tmem_lifecycle
  // CHECK: tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: tcgen05.dealloc.cta_group::2.sync.aligned.b32
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @automatic_tmem_lifecycle() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 256>} {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = nvg.tensor_memory_base
      %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
      "use"(%1) : (i32) -> ()
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

// TLX paired-CTA MMA (2-CTA) warp-specialized kernel: the compiler inserts a
// cluster arrive/wait before TMEM dealloc (baseline for the skip below).
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.tensor_memory_size = 128 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: @tmem_lifecycle_paired_mma
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: tcgen05.dealloc.cta_group::2.sync.aligned.b32
  llvm.func @tmem_lifecycle_paired_mma() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 256>} {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = nvg.tensor_memory_base
      %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
      "use"(%1) : (i32) -> ()
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

// Same kernel but with tlx.user_post_ws_sync: the user handles the
// post-warp-specialize sync, so the compiler skips the cluster arrive/wait
// before TMEM dealloc (tcgen05 alloc/dealloc unchanged).
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.tensor_memory_size = 128 : i32, tlx.user_post_ws_sync = true} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: @tmem_lifecycle_user_post_ws_sync
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
  // CHECK-NOT: nvvm.cluster.arrive
  // CHECK-NOT: nvvm.cluster.wait
  // CHECK: tcgen05.dealloc.cta_group::2.sync.aligned.b32
  llvm.func @tmem_lifecycle_user_post_ws_sync() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 256>} {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = nvg.tensor_memory_base
      %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
      "use"(%1) : (i32) -> ()
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}
