
// RUN: triton-opt -split-input-file --verify-diagnostics %s | FileCheck %s

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @require_layout
  tt.func @require_layout(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem>) {
    // CHECK: tlx.require_layout
    %0 = tlx.require_layout %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem> -> !ttg.memdesc<128x64xf16, #shared2, #smem>
    tt.return
  }
}

// -----

#one_per_lane = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @warp_votes
  tt.func @warp_votes(%pred: tensor<64xi1, #one_per_lane>) -> (i1, i1) {
    // CHECK: %[[ALL:.*]] = ttg.warp_vote %{{.*}} "all"
    %all = ttg.warp_vote %pred "all" : tensor<64xi1, #one_per_lane> -> i1
    // CHECK: %[[ANY:.*]] = ttg.warp_vote %{{.*}} "any"
    %any = ttg.warp_vote %pred "any" : tensor<64xi1, #one_per_lane> -> i1
    tt.return %all, %any : i1, i1
  }
}

// -----

#one_per_lane = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_vote_rejects_unknown_kind(%pred: tensor<64xi1, #one_per_lane>) -> i1 {
    // expected-error @+1 {{'ttg.warp_vote' op kind must be "all" or "any"}}
    %vote = ttg.warp_vote %pred "some" : tensor<64xi1, #one_per_lane> -> i1
    tt.return %vote : i1
  }
}

// -----

#two_per_lane = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_vote_rejects_multiple_elements(%pred: tensor<128xi1, #two_per_lane>) -> i1 {
    // expected-error @+1 {{'ttg.warp_vote' op predicate must distribute exactly one element per lane}}
    %vote = ttg.warp_vote %pred "all" : tensor<128xi1, #two_per_lane> -> i1
    tt.return %vote : i1
  }
}
