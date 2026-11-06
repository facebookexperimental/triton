// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file | FileCheck %s

// TLX maps a barrier to a peer CTA itself: barrier_arrive takes the local
// barrier plus remote_cta_rank and rejects a remote_view, and a two-CTA
// async_descriptor_load maps its barrier to the leader CTA. The TTGIR already
// carries the ttng.map_to_remote_buffer, so it is folded into those ops rather
// than printed as a tlx.remote_view they would refuse.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def remote_arrive(
  // CHECK: [[BAR:[a-z_0-9]+]] = tlx.alloc_barriers(1)
  // CHECK: [[RANK:[a-z_0-9]+]] = tlx.cluster_cta_rank()
  // CHECK-NOT: tlx.remote_view
  // CHECK: tlx.barrier_arrive([[BAR]], remote_cta_rank={{[a-z_0-9]+}}, pred=
  tt.func public @remote_arrive() attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %rank = nvg.cluster_id
    %peer = arith.xori %rank, %c1_i32 : i32
    %parity = arith.remui %rank, %c2_i32 : i32
    %leader = arith.cmpi eq, %parity, %c0_i32 : i32
    %remote = ttng.map_to_remote_buffer %bar, %peer : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote, 1, %leader : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: def two_cta_load(
  // CHECK: [[LBAR:[a-z_0-9]+]] = tlx.alloc_barriers(1)
  // CHECK-NOT: tlx.remote_view
  // CHECK: tlx.async_descriptor_load({{.*}}, [[LBAR]], two_ctas=True)
  tt.func public @two_cta_load(%desc: !tt.tensordesc<128x128xbf16, #shared1>, %x: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cm2_i32 = arith.constant -2 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    %rank = nvg.cluster_id
    %leader_rank = arith.andi %rank, %cm2_i32 : i32
    %remote = ttng.map_to_remote_buffer %bar, %leader_rank : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %buf, %remote, %true {two_cta = true} : !tt.tensordesc<128x128xbf16, #shared1>, !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
    tt.return
  }
}
