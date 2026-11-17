// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s

// Nothing after a single-WS warp_specialize op touches memory, so it is not
// marked for a join, and the barrier invalidations after it are dropped: they
// would otherwise run while a partition may still wait on the barrier.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @no_join_without_post_ws_memory_ops
// CHECK-NOT: ttg.warp_specialize.join
// CHECK: } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
// CHECK-NOT: ttng.inval_barrier
// CHECK: tt.return
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @no_join_without_post_ws_memory_ops() {
  %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
  ttg.warp_specialize(%bar)
  default {
    %c0 = arith.constant {ttg.warp_specialize.tag = 0 : i32} 0 : i32
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(4) {
    ttg.warp_return
  } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
  ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  tt.return
}

}

// -----

// A load after the warp_specialize op reads what the partitions wrote, so the
// op is marked for a join, the invalidations stay, and the partition that
// issues TMA stores drains them before it ends.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @join_before_post_ws_load
// CHECK: ttg.warp_specialize.join
// CHECK: default {
// CHECK-NOT: ttng.async_tma_store_wait
// CHECK: ttg.warp_yield
// CHECK: partition0
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
// CHECK-NEXT: ttg.warp_return
// CHECK: ttng.inval_barrier
// CHECK: tt.load
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @join_before_post_ws_load(%desc: !tt.tensordesc<128x64xf16, #nvmma>, %ptr: !tt.ptr<f32>) {
  %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
  ttg.warp_specialize(%bar, %desc, %buf)
  default {
    %c0 = arith.constant {ttg.warp_specialize.tag = 0 : i32} 0 : i32
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %arg1: !tt.tensordesc<128x64xf16, #nvmma>, %arg2: !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) num_warps(4) {
    %c0_i32 = arith.constant 0 : i32
    ttng.async_tma_copy_local_to_global %arg1[%c0_i32, %c0_i32] %arg2 : !tt.tensordesc<128x64xf16, #nvmma>, !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>, !tt.tensordesc<128x64xf16, #nvmma>, !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) -> ()
  ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  %v = tt.load %ptr : !tt.ptr<f32>
  tt.return
}

}

// -----

// The TMA store is issued by a noinline helper the partition calls, so the
// partition still drains it before it ends.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @join_drains_tma_store_in_noinline_helper
// CHECK: ttg.warp_specialize.join
// CHECK: partition0
// CHECK: tt.call @tma_store_helper
// CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
// CHECK-NEXT: ttg.warp_return
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func private @tma_store_helper(%desc: !tt.tensordesc<128x64xf16, #nvmma>, %buf: !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) attributes {noinline = true} {
  %c0_i32 = arith.constant 0 : i32
  ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32] %buf : !tt.tensordesc<128x64xf16, #nvmma>, !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
  tt.return
}

tt.func @join_drains_tma_store_in_noinline_helper(%desc: !tt.tensordesc<128x64xf16, #nvmma>, %ptr: !tt.ptr<f32>) {
  %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
  ttg.warp_specialize(%desc, %buf)
  default {
    %c0 = arith.constant {ttg.warp_specialize.tag = 0 : i32} 0 : i32
    ttg.warp_yield
  }
  partition0(%arg0: !tt.tensordesc<128x64xf16, #nvmma>, %arg1: !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) num_warps(4) {
    tt.call @tma_store_helper(%arg0, %arg1) : (!tt.tensordesc<128x64xf16, #nvmma>, !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) -> ()
    ttg.warp_return
  } : (!tt.tensordesc<128x64xf16, #nvmma>, !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>) -> ()
  %v = tt.load %ptr : !tt.ptr<f32>
  tt.return
}

}
