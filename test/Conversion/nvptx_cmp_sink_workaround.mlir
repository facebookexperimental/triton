// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' | FileCheck %s --check-prefixes=CHECK,SM107
// The cluster case is checked only on sm_107, the one target with multicast
// arrives (PTX 9.4 or later).

// `tlx.async_task(nvptx_cmp_sink_workaround=True)` is a TEMPORARY workaround
// for NVPTX letting CodeGenPrepare sink compares into the blocks of their
// users. The TLX code generator lists the flagged partitions on the
// warp_specialize op. In those partitions only:
//  - a per-warp arrive tests %laneid inside its own asm, so no lane predicate
//    is kept live across loops;
//  - leader arrives share one predicate, built at the partition entry by a
//    side-effecting setp that CodeGenPrepare cannot sink.
// Arrives in other partitions lower as usual.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 12 : i32} {
  // CHECK-LABEL: @nvptx_cmp_sink_workaround
  tt.func @nvptx_cmp_sink_workaround(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %pred: i1) {
    ttg.warp_specialize(%bar, %pred) attributes {tlx.nvptx_cmp_sink_workaround = array<i32: 0>, warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    // The flagged partition builds its leader predicate first, at the entry.
    // CHECK: partition0
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: [[LEADER:%.*]] = llvm.inline_asm has_side_effects {{.*}}"setp.eq.u32 $0, $1, 0;", "=b,r"
    // CHECK-NOT: llvm.icmp "eq"
    // CHECK: "{ .reg .pred lane0; .reg .b32 lane; mov.u32 lane, %laneid; setp.eq.u32 lane0, lane, 0; @lane0 mbarrier.arrive.shared::cta.b64 _, [$1]; }"
    // CHECK: "{ .reg .pred lane0; .reg .b32 lane; mov.u32 lane, %laneid; setp.eq.u32 lane0, lane, 0; and.pred lane0, lane0, $0; @lane0 mbarrier.arrive.shared::cta.b64 _, [$1]; }"
    // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r" [[LEADER]]
    // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r" [[LEADER]]
    // CHECK-NOT: setp.eq.u32
    // CHECK: ttg.warp_return
    partition0(%b0: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %p0: i1) num_warps(4) {
      ttng.arrive_barrier %b0, 4 {per_warp} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %b0, 4, %p0 {per_warp} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %b0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %b0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    }
    // An unflagged partition keeps the usual compares.
    // CHECK: partition1
    // CHECK-NOT: %laneid
    // CHECK-NOT: setp.eq.u32
    // CHECK: llvm.icmp "eq"
    // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];"
    // CHECK: llvm.icmp "eq"
    // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];"
    // CHECK: ttg.warp_return
    partition1(%b1: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %p1: i1) num_warps(4) {
      ttng.arrive_barrier %b1, 4 {per_warp} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %b1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>, i1) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // The code generator lists the default task as -1.
  // CHECK-LABEL: @nvptx_cmp_sink_workaround_default
  tt.func @nvptx_cmp_sink_workaround_default(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // CHECK: ttg.warp_specialize
    // CHECK-NEXT: default
    // CHECK: [[LEADER:%.*]] = llvm.inline_asm has_side_effects {{.*}}"setp.eq.u32 $0, $1, 0;", "=b,r"
    // CHECK-NOT: llvm.icmp "eq"
    // CHECK: mov.u32 lane, %laneid; setp.eq.u32 lane0, lane, 0; @lane0 mbarrier.arrive
    // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r" [[LEADER]]
    // CHECK: ttg.warp_yield
    // CHECK: partition0
    // CHECK-NOT: %laneid
    // CHECK-NOT: setp.eq.u32
    // CHECK: llvm.icmp "eq"
    // CHECK: ttg.warp_return
    ttg.warp_specialize(%bar) attributes {tlx.nvptx_cmp_sink_workaround = array<i32: -1>, warpGroupStartIds = array<i32: 4>}
    default {
      ttng.arrive_barrier %bar, 4 {per_warp} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%b0: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(4) {
      ttng.arrive_barrier %b0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// Cluster-scope arrives in a flagged partition.
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2], [4]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0], [0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:107"} {
  // SM107-LABEL: @nvptx_cmp_sink_workaround_cluster
  tt.func @nvptx_cmp_sink_workaround_cluster(%bar: !ttg.memdesc<8xi64, #barrier, #smem, mutable>, %remote: !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>) {
    ttg.warp_specialize(%bar, %remote) attributes {tlx.nvptx_cmp_sink_workaround = array<i32: 0>, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    // SM107: partition0
    // SM107: [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // SM107: [[LEADER:%.*]] = llvm.inline_asm has_side_effects {{.*}}"setp.eq.u32 $0, $1, 0;", "=b,r"
    // SM107-NOT: llvm.icmp "eq"
    // A per-warp arrive on a cluster-memory barrier keeps cluster scope. With
    // no user predicate, $0 is bound to true but not referenced.
    // SM107: "{ .reg .pred lane0; .reg .b32 lane; mov.u32 lane, %laneid; setp.eq.u32 lane0, lane, 0; @lane0 mbarrier.arrive.shared::cluster.b64 _, [$1]; }", "b,r" [[TRUE]], %{{.*}} : (i1, !llvm.ptr<7>)
    // A per-warp multicast arrive: the mask is $2, next to the unused $0.
    // SM107: "{ .reg .pred lane0; .reg .b32 lane; mov.u32 lane, %laneid; setp.eq.u32 lane0, lane, 0; @lane0 mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64 _, [$1], $2; }", "b,r,r" [[TRUE]], %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>, i32)
    // A multicast leader arrive uses the entry predicate as $0 and the mask as $2.
    // SM107: "@$0 mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64 _, [$1], $2;", "b,r,r" [[LEADER]], %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>, i32)
    // SM107-NOT: llvm.icmp "eq"
    // SM107-NOT: setp.eq.u32 $0
    // SM107: ttg.warp_return
    partition0(%b0: !ttg.memdesc<8xi64, #barrier, #smem, mutable>, %r0: !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>) num_warps(4) {
      ttng.arrive_barrier %r0, 4 {per_warp} : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %b0, 4 {multicastCTA = 2 : i32, per_warp} : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %b0, 1 {multicastCTA = 2 : i32} : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<8xi64, #barrier, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>) -> ()
    tt.return
  }
}
