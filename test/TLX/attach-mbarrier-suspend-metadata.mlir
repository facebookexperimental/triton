// RUN: triton-opt %s -split-input-file --triton-tlx-fixup="target=cuda:100" | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: tlx.mbarrier_try_wait_suspend_ns = 30000 : i32
module {
  tt.func @minimum_explicit_value() {
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 50000 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 30000 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    tt.return
  }
}

// -----

// CHECK: module attributes {
// CHECK-SAME: tlx.mbarrier_try_wait_suspend_ns = 50000 : i32
module {
  tt.func @unspecified_does_not_participate() {
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 50000 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    ttg.warp_specialize()
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    tt.return
  }
}

// -----

// CHECK: module attributes {
// CHECK-SAME: tlx.mbarrier_try_wait_suspend_ns = 0 : i32
module {
  tt.func @explicit_zero_wins() {
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 50000 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    ttg.warp_specialize() attributes {tlx.mbarrier_try_wait_suspend_ns = 0 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return } : () -> ()
    tt.return
  }
}
