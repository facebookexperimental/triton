// RUN: triton-opt -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm --verify-diagnostics %s| FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tlx_bar_init
  tt.func public @tlx_bar_init() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %2 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %2, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tlx_bar_init_ws_partition
  tt.func public @tlx_bar_init_ws_partition() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%0) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(1) {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32_0 = arith.constant 0 : i32
      %7 = ttg.memdesc_index %arg3[%c0_i32_0] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %8 = ttng.map_to_remote_buffer %7, %c0_i32_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %8, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tlx_bar_init_ws_default
  tt.func public @tlx_bar_init_ws_default() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize()
    default {
      %true = arith.constant true
      %false = arith.constant false
      %c0_i32_0 = arith.constant 0 : i32
      %7 = ttg.memdesc_index %0[%c0_i32_0] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %8 = ttng.map_to_remote_buffer %7, %c0_i32_0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %8, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tlx_bar_init_for_block
  tt.func public @tlx_bar_init_for_block() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c1_i32 = arith.constant 1 : i32
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK-NEXT: nvvm.cluster.arrive {aligned}
    // CHECK-NEXT: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %2, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0_i32_0 = arith.constant 0 : i32
    %c300_i32 = arith.constant 300 : i32
    %c1_i32_1 = arith.constant 1 : i32

    scf.for %arg6 = %c0_i32_0 to %c300_i32 step %c1_i32_1  : i32 {
      %8 = ttg.memdesc_index %0[%arg6] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %c0_i32_3 = arith.constant 0 : i32
      %9 = ttng.map_to_remote_buffer %8, %c0_i32_3 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %9, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    }
    %true = arith.constant true
    %false = arith.constant false
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @tlx_bar_init_ws_non_first_block() attributes {noinline = false} {
    ttg.warp_specialize()
    default {
      %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %c0_i32 = arith.constant 0 : i32
      %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      // expected-error @+1 {{Barrier init outside of the first block in function is not supported for remote barriers}}
      ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %9 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = tt.get_program_id x : i32
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// Test that cluster sync is inserted after barrier init for clustered kernels
// using cluster-dim-x (without paired CTA MMA attribute).
// This exercises the tlxIsClustered API for cluster sync insertion.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @clustered_bar_init_sync
  tt.func public @clustered_bar_init_sync() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %2 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %2, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_commit with {two_ctas} triggers cluster sync after init_barrier.
// The two_ctas flag means the barrier signal is multicast to other CTAs in the cluster.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_commit_two_ctas_bar_init
  tt.func public @tc_gen5_commit_two_ctas_bar_init() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: tcgen05.commit
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tc_gen5_commit %1 {two_ctas} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that async_clc_try_cancel triggers cluster sync after init_barrier.
// The CLC try_cancel always multicasts the barrier signal to all CTAs in the cluster.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @clc_try_cancel_bar_init
  tt.func public @clc_try_cancel_bar_init() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1xui128, #shared, #smem, mutable>
    ttng.async_clc_try_cancel %1, %2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xui128, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that async_tma_copy_global_to_local with multicast_targets triggers cluster
// sync after init_barrier. The multicast bitmask causes the barrier signal to be
// sent to multiple CTAs in the cluster.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tma_multicast_bar_init
  tt.func public @tma_multicast_bar_init(%desc: !tt.tensordesc<tensor<128x64xbf16, #nvmma>>, %alloc: !ttg.memdesc<128x64xbf16, #nvmma, #smem, mutable>, %x: i32, %mcast: i32, %pred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %1, %pred, %mcast : !tt.tensordesc<tensor<128x64xbf16, #nvmma>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_commit WITHOUT {two_ctas} does NOT trigger cluster sync.
// The barrier signal stays local, so no cluster bootstrap is needed.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_commit_no_two_ctas_no_sync
  tt.func public @tc_gen5_commit_no_two_ctas_no_sync() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tc_gen5_commit %1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that async_tma_copy_global_to_local WITHOUT multicast_targets does NOT
// trigger cluster sync, even in a clustered kernel.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tma_no_multicast_no_sync
  tt.func public @tma_no_multicast_no_sync(%desc: !tt.tensordesc<tensor<128x64xbf16, #nvmma>>, %alloc: !ttg.memdesc<128x64xbf16, #nvmma, #smem, mutable>, %x: i32, %pred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %1, %pred : !tt.tensordesc<tensor<128x64xbf16, #nvmma>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// Test that tmem_copy with barrier in paired CTA MMA mode triggers cluster sync.
// The barrier on tmem_copy will generate a tcgen05.commit with multicast in 2cta mode.
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_scales = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]]}, alignment = 16>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @tmem_copy_barrier_paired_cta
  tt.func public @tmem_copy_barrier_paired_cta(
      %src: !ttg.memdesc<128x32xi8, #shared_scales, #ttg.shared_memory>,
      %dst: !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: tcgen05.cp
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tmem_copy %src, %dst, %1 : !ttg.memdesc<128x32xi8, #shared_scales, #ttg.shared_memory>, !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test that tmem_copy with barrier but WITHOUT paired CTA MMA does NOT trigger
// cluster sync. The commit stays local without multicast.
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_scales = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]]}, alignment = 16>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tmem_copy_barrier_no_paired_cta_no_sync
  tt.func public @tmem_copy_barrier_no_paired_cta_no_sync(
      %src: !ttg.memdesc<128x32xi8, #shared_scales, #ttg.shared_memory>,
      %dst: !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tmem_copy %src, %dst, %1 : !ttg.memdesc<128x32xi8, #shared_scales, #ttg.shared_memory>, !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_mma with multiple barriers in paired CTA MMA mode triggers
// cluster sync. The MMA's commit will multicast barrier signals to other CTAs
// in 2cta mode. Both barriers must be initialized before the cluster sync.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_barrier_paired_cta
  tt.func public @tc_gen5_mma_barrier_paired_cta(
      %a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x256xf16, #tmem, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1, %barrierPred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<2xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: tcgen05.mma
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %1[%barrierPred], %2[%barrierPred] {is_async, two_ctas} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>,
       !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_mma with barrier but WITHOUT paired CTA MMA does NOT trigger
// cluster sync, even in a clustered kernel.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_barrier_no_paired_cta_no_sync
  tt.func public @tc_gen5_mma_barrier_no_paired_cta_no_sync(
      %a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x256xf16, #tmem, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1, %barrierPred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %1[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_mma_scaled with barrier in paired CTA MMA mode triggers
// cluster sync. The scaled MMA's commit multicasts barrier signals in 2cta mode.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_scaled_barrier_paired_cta
  tt.func public @tc_gen5_mma_scaled_barrier_paired_cta(
      %a: !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<64x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %useAcc: i1, %pred: i1, %barrierPred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: tcgen05.mma
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e4m3, %1[%barrierPred] {is_async, two_ctas} :
       !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>,
       !ttg.memdesc<64x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_mma_scaled with barrier but WITHOUT paired CTA MMA does NOT
// trigger cluster sync, even in a clustered kernel.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_scaled_barrier_no_paired_cta_no_sync
  tt.func public @tc_gen5_mma_scaled_barrier_no_paired_cta_no_sync(
      %a: !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<64x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %useAcc: i1, %pred: i1, %barrierPred: i1) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e4m3, %1[%barrierPred] {is_async} :
       !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>,
       !ttg.memdesc<64x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}
