// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' | FileCheck %s

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
