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
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
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
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tlx_bar_init_ws_default
  tt.func public @tlx_bar_init_ws_default() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
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
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK-NEXT: nvvm.cluster.wait {aligned}
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
      // expected-error @+1 {{Barrier init outside of the first block in function is not supported}}
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
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
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

// Test that tc_gen5_commit with two_ctas triggers cluster sync after init_barrier.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @tc_gen5_commit_two_ctas_bar_init
  tt.func public @tc_gen5_commit_two_ctas_bar_init() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tc_gen5_commit %1 {two_ctas} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that async_clc_try_cancel triggers cluster sync after init_barrier.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @clc_try_cancel_bar_init
  tt.func public @clc_try_cancel_bar_init() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
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
// sync after init_barrier.
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
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK: nvvm.cluster.wait {aligned}
    // CHECK: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %1, %pred, %mcast : !tt.tensordesc<tensor<128x64xbf16, #nvmma>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

// Test that tc_gen5_commit WITHOUT two_ctas does NOT trigger cluster sync.
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

// Test that fence.proxy.async.shared::cluster is inserted after
// fence.mbarrier_init for all clustered kernels (not just paired MMA).
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @fence_proxy_async_all_clustered
  tt.func public @fence_proxy_async_all_clustered() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
    // CHECK: nvvm.cluster.arrive {aligned}
    // CHECK-NEXT: nvvm.cluster.wait {aligned}
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %2 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %2, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}

// -----

// Test that explicit_cluster_sync suppresses heuristic cluster sync insertion.
// Even though there is a remote barrier (map_to_remote_buffer + arrive_barrier),
// the compiler must not auto-insert cluster arrive/wait because the user is
// responsible for placing them manually.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, tlx.explicit_cluster_sync = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @explicit_cluster_sync_no_auto_insert
  tt.func public @explicit_cluster_sync_no_auto_insert() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK-NOT: nvvm.cluster.arrive
    // CHECK-NOT: nvvm.cluster.wait
    // CHECK: nvvm.mapa
    ttng.init_barrier %1, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %2 = ttng.map_to_remote_buffer %1, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %2, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}
