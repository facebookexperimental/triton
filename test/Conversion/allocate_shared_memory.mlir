// RUN: triton-opt %s -split-input-file --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked>, tensor<1024x256xi32, #blocked>) -> tensor<1024x256xf32, #blocked>
  tt.return
}

}

// -----

// Single-warp-specialize dispatches worker warps from `wid`, so the per-warp
// state-id array is not allocated -- this module needs no shared memory.
// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.single-warp-specialize" = true} {

// CHECK-LABEL: @single_ws_no_state_smem
// CHECK-NOT: allocation.offset
tt.func @single_ws_no_state_smem() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// Without the attribute, an i8 per worker warp is reserved for the state id
// (num_warps = 4 -> 4 bytes).
// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 4 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @ws_state_smem
tt.func @ws_state_smem() {
  // CHECK: allocation.offset = 0 : i32
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
